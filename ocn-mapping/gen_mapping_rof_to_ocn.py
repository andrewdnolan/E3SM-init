#!/usr/bin/env python3

import click
import logging
import netCDF4
import numpy as np
import re
import shutil
import tempfile
import xarray as xr

from datetime import datetime
from enum import Enum, auto
from logging import Logger
from mache import MachineInfo, discover_machine
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from pathlib import Path
from typing import Literal, NoReturn, Optional
from xarray.core.dataarray import DataArray

click_input_path = click.Path(
    exists=True, dir_okay=False, readable=True, path_type=Path
)

class FileType(Enum):
    DAITREN = auto()
    MOSART = auto()
    RTM = auto()
    SCRIP = auto()

def detect_file_type(path: Path) -> FileType:
    """
    Detect the type (daitren/mosart/rtm/scrip) of the input file.

    NOTE: Currently this soely does this based off the filename.
    It does not actually check that the infered file type is properly
    formatted, by checking dimensions and/or attributes.

    Parameters
    ----------
    fp: pathlib.Path
        Path to the input file

    Returns
    -------
    FileType
        Enum'ed file type
    """
    stem = str(path.stem).lower()

    if "scrip" in stem:
        return FileType.SCRIP
    elif "domain" in stem:
        return FileType.DOMAIN
    elif "daitren" in stem:
        return FileType.RTM
    elif "mosart" in stem:
        return FileType.MOSART
    else:
        raise ValueError()

def calculate_spherical_quad_area(xv: DataArray, yv: DataArray) -> DataArray:
    """
    Calculate the area of a quadrilateral grid cell on a sphere

    Parameters
    ----------
    xv : xr.DataArray
        x vertex coordinate array in degrees

    yv : xr.DataArray
        y vertex coordinate arrays in degrees
    """

    def calc_area(lat_1, lat_2, lon_1, lon_2):
        """Eqn (1) from Kelly and Savric (2021)
        """
        return np.abs(lon_2 - lon_1) * (np.sin(lat_2) - np.sin(lat_1))

    if xv.attrs['units'].lower() != 'degrees':
        raise ValueError("units wrong")

    xv = np.deg2rad(xv)
    yv = np.deg2rad(yv)

    #TODO: Explain why this works with counter clockwise ordering
    return calc_area(yv[:, 0], yv[:, 2], xv[:, 0], xv[:, 1])

def generate_mapping_file(
    rof_scrip: Path,
    ocn_scrip: Path,
    weight_fn: Path,
    logger: Logger,
    parallel_executable: Optional[str] = None,
    nprocs: int = 1) -> NoReturn:
    """
    Generate the nearest neighbor mapping file

    Parameters
    ----------
    rof_scrip : str
        filepath to the SCRIP file describing the rof grid

    ocn_scrip : str
        filepath to the SCRIP file describing the ocn grid

    weight_fn: str
        filename of the resulting mapping file

    parallel_executable : str or None
        executable to use for a parallel job. If None, then run in serial

    nprocs : int
        number of processors to use for generating remapping weights
    """

    # Generate remapping weights
    logger.info('generating rof -> mpaso nearest neighbor weights')

    args = []

    if (nprocs > 1) & (parallel_executable != None):
        args += [parallel_executable, '-n', str(nprocs)]

    args += ['ESMF_RegridWeightGen',
             '--source', str(rof_scrip),
             '--destination', str(ocn_scrip),
             '--weight', str(weight_fn),
             '--method', 'nearestdtos',
             '--netcdf4',]

    check_call(args, logger=logger)

def area_weight_mapping_file(weight_fn: Path, logger: Logger) -> NoReturn:
    """
    Area weight the nearest neighbor mapping file

    Parameters
    ----------
    weight_fn : str
        filepath to the weight file to area weighted
    """
    ds = xr.open_dataset(weight_fn)

    if np.all(ds.area_a == 0.):
        logger.info('\"area_a\" is missing from mapping file. Calculating...')
        ds['area_a'] = calculate_spherical_quad_area(ds.xv_a, ds.yv_a)
        ds['area_a'].attrs["units"] = "square radians"

    if np.all(ds.area_b == 0.):
        raise ValueError()

    ds['S'] = ds.area_a[ds.col - 1] / ds.area_b[ds.row - 1]

    write_netcdf(ds, weight_fn)

def convert_to_scrip(rof_file: Path) -> Path:
    """
    Convert input file (daitren/mosart/rtm) to the SCRIP format.

    Parameters
    ----------
    rof_file : pathlib.Path
        path of a domain or rtm data file to be converted

    Returns
    -------
    rof_scrip : pathlib.Path
        path to the temporary SCRIP file
    """

    def fmt_attrs(da: DataArray) -> DataArray:
        """
        Format the 'units' attribute for the SCRIP coordinate arrays

        NOTE: This is only valid for coordinates arrays. This will not format
        the units of the 'grid_imask' or 'grid_area' arrays correctly.

        Parameters
        ----------
        da: xr.DataArray
            coordinate array to parse and format the unit attribute of
        """
        # if units of coordinate array can not be verified, then raise error
        if 'units' not in da.attrs:
            raise ValueError()

        if re.search("degree", da.attrs["units"]):
            units = "degrees"
        elif re.search("radian", da.attrs["units"]):
            units = "radians"
        else:
            raise ValueError()

        # wipe the existing
        da = da.drop_attrs()
        # add "unit" attribute following SCRIP format
        da.attrs["units"] = units

        return da

    rof_ds = xr.open_dataset(rof_file)
    rof_file_type = detect_file_type(rof_file)

    if rof_file_type in (FileType.DAITREN, FileType.RTM):
        grid_dims = np.array((rof_ds.sizes["nj"], rof_ds.sizes["ni"]))
    elif rof_file_type is FileType.MOSART:
        grid_dims = np.array((rof_ds.sizes["lat"], rof_ds.sizes["lon"]))
    else:
        raise ValueError()

    scrip_ds = xr.Dataset({
        "grid_dims": xr.DataArray(grid_dims, dims="grid_rank")
    })

    if rof_file_type in (FileType.DAITREN, FileType.RTM):
        rof_ds = rof_ds.stack(grid_size=("nj", "ni"), create_index=False)
        rof_ds = rof_ds.rename(nv="grid_corners")

        # rtm data files have "nt" dimension
        if "nt" in rof_ds.dims:
            rof_ds = rof_ds.drop_dims("nt")

        # ensure the dimensions are order as needed by the SCRIP format
        rof_ds = rof_ds.transpose("grid_size", "grid_corners")

        if ('xc' in rof_ds.coords) and ('yc' in rof_ds.coords):
            rof_ds = rof_ds.reset_coords(("xc", "yc"))

        grid_center_lat = fmt_attrs(rof_ds["yc"])
        grid_center_lon = fmt_attrs(rof_ds["xc"])
        grid_corner_lat = fmt_attrs(rof_ds["yv"])
        grid_corner_lon = fmt_attrs(rof_ds["xv"])

    elif rof_file_type is FileType.MOSART:
        lat_2d = rof_ds.latixy.values
        lon_2d = rof_ds.longxy.values

        delta_lat = lat_2d[1:, :] - lat_2d[:-1, :]
        delta_lon = lon_2d[:, 1:] - lon_2d[:, :-1]

        if not np.allclose(delta_lat, delta_lat[0, 0], rtol=0, atol=1e-12):
            raise AssertionError("Lattitude spacing is NOT constant")

        if not np.allclose(delta_lon, delta_lon[0, 0], rtol=0, atol=1e-12):
            raise AssertionError("Longitude spacing is NOT constant")

        dlat = delta_lon[0, 0] / 2.
        dlon = delta_lat[0, 0] / 2.

        grid_center_lat = xr.DataArray(
            lat_2d.flatten(), dims="grid_size", attrs={"units": "degrees"}
        )
        grid_center_lon = xr.DataArray(
            lon_2d.flatten(), dims="grid_size", attrs={"units": "degrees"}
        )

        grid_corner_lat = grid_center_lat.expand_dims(
            dim={"grid_corners": 4}, axis=1
        )
        grid_corner_lon = grid_center_lon.expand_dims(
            dim={"grid_corners": 4}, axis=1
        )

        grid_corner_lon = grid_corner_lon + dlon * np.array([[-1, 1, 1, -1]])
        grid_corner_lat = grid_corner_lat + dlat * np.array([[-1, -1, 1, 1]])
        # after the broadcasting attrs are wipped out, so manually reset
        grid_corner_lon.attrs["units"] = "degrees"
        grid_corner_lat.attrs["units"] = "degrees"

    scrip_ds["grid_center_lon"] = grid_center_lon
    scrip_ds["grid_center_lat"] = grid_center_lat
    scrip_ds["grid_corner_lon"] = grid_corner_lon
    scrip_ds["grid_corner_lat"] = grid_corner_lat

    scrip_ds['grid_area'] = calculate_spherical_quad_area(
        scrip_ds.grid_corner_lon, scrip_ds.grid_corner_lat
    )
    scrip_ds['grid_area'].attrs["units"] = "square radians"
    scrip_ds['grid_imask'] = xr.ones_like(grid_center_lon).drop_attrs()

    # create the filepath to a temporary scrip file
    tmp_dir = tempfile.mkdtemp()
    tmp_fn = rof_file.stem + f".scrip.nc"
    tmp_rof_scrip = Path(tmp_dir) / tmp_fn

    write_netcdf(scrip_ds, tmp_rof_scrip)

    return tmp_rof_scrip

def mask_rof_scrip(
    rof_scrip: Path, rof_mesh: Path, mask_var: str, lnd_domain: Path
) -> Path:
    """
    Mask the rof scrip file based on the MOSART variable requested and
    the union of the lnd domain file

    Parameters
    ----------
    rof_scrip : pathlib.Path
        path to the rof SCRIP file to be masked

    rof_mesh : pathlib.Path
        path the rof mesh file containing the variable to be used to generate
        the mask

    mask_var : str
        variable name to generate the mask from

    lnd_domain : pathlib.Path
        path to the lnd domain file used for masking

    Returns
    -------
    rof_scrip : pathlib.Path
        Path the temporary SCRIP file with an updated `grid_imask` field
    """
    rof_ds = xr.open_dataset(rof_mesh)
    lnd_ds = xr.open_dataset(lnd_domain)
    scrip_ds = xr.open_dataset(rof_scrip)

    if mask_var not in rof_ds:
        raise ValueError("{mask_var} can not be found in {rof_mesh}")

    # get the integer type of the original scrip file
    dtype = rof_ds[mask_var].dtype
    # get a mask of where rof field is *not* nan
    rof_mask = (~np.isnan(rof_ds[mask_var].values))
    # get the lnd domain mask and convert to booleans
    lnd_mask = lnd_ds.mask.astype(bool).values
    # take the union of the rof and lnd masks
    imask = (rof_mask | lnd_mask).flatten().astype(dtype)

    # have to set the values b/c `imask` is a np.ndarray
    scrip_ds["grid_imask"].values = imask

    # if input is a tmp scrip, then place the masked scrip in the same tmp dir
    if re.search("^/tmp/tmp", str(rof_scrip)):
        tmp_dir = rof_scrip.parent
    else:
        # create the filepath to a temporary scrip file
        tmp_dir = Path(tempfile.mkdtemp())

    tmp_fn = rof_scrip.stem + f".CUSTOM_{mask_var}_MASK.nc"
    tmp_rof_scrip = tmp_dir / tmp_fn

    write_netcdf(scrip_ds, tmp_rof_scrip)

    return tmp_rof_scrip

def add_grid_size_dimensions(
    weights_fn: Path, rof_scrip: Path, ocn_scrip: Path
) -> NoReturn:
    """
    Add the missing dimensions cpl6 needs to decompose the mapping files

    netCDF4 has to be used directly here, becuase xarray does not support
    dimensions that are not used by any of dataarrays within the dataset

    Parameters
    ----------
    weights_fn : str
        filepath to the weight file to area weighted

    rof_scrip : str
        filepath to the SCRIP file describing the rof grid

    ocn_scrip : str
        filepath to the SCRIP file describing the ocn grid
    """
    rof_ds = xr.open_dataset(rof_scrip)
    ocn_ds = xr.open_dataset(ocn_scrip)

    nj_a, ni_a = rof_ds.grid_dims.values
    nj_b, ni_b = 1, ocn_ds.sizes["grid_size"]

    # have to use netCDF4 b/c xarray does not allow dims not used by a field
    with netCDF4.Dataset(weights_fn, "r+") as weights_ds:
        weights_ds.createDimension("nj_a", nj_a)
        weights_ds.createDimension("ni_a", ni_a)
        weights_ds.createDimension("nj_b", nj_b)
        weights_ds.createDimension("ni_b", ni_b)

def create_logger() -> Logger:
    """
    Create a logger and stream output to file
    """
    script_stem = "map_rof_to_ocn"
    datetime_str = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')

    logger_fn = script_stem + "_" + datetime_str + ".log"

    logger = logging.getLogger(" ")
    logging.basicConfig(
        filename=logger_fn, encoding='utf-8', level=logging.INFO
    )

    return logger

@click.command()
@click.option('-r', '--rof_file', type=click_input_path)
@click.option('-o', '--ocn_file', type=click_input_path)
@click.option('-w', '--weight_fn', type=Path)
@click.option('-n', '--nprocs', type=int, default=1)
@click.option('--mask_var', type=str)
@click.option('--lnd_domain', type=Path)
def gen_mapping_rof_to_ocn(
    rof_file, ocn_file, weight_fn, nprocs, mask_var, lnd_domain
):
    """
    Generate an area weighted nearest neighbor mapping file
    """

    logger = create_logger()

    if mask_var != None and lnd_domain == None:
        raise ValueError()

    if mask_var != None and detect_file_type(rof_file) is not FileType.MOSART:
        raise ValueError(
            "Masking func, which checks for NaN, only works for MOSART files"
        )

    if detect_file_type(ocn_file) is not FileType.SCRIP:
        raise ValueError()

    if detect_file_type(rof_file) is not FileType.SCRIP:
        rof_scrip = convert_to_scrip(rof_file)

        if mask_var != None:
            rof_scrip = mask_rof_scrip(
                rof_scrip, rof_file, mask_var, lnd_domain
            )

    parallel_executable = None
    if nprocs > 1:
        machine = discover_machine()
        config = MachineInfo(machine).config

        parallel_executable = config.get("parallel", "parallel_executable")

    generate_mapping_file(
        rof_scrip, ocn_file, weight_fn, logger, parallel_executable, nprocs
    )

    area_weight_mapping_file(weight_fn, logger)

    # NOTE: this has to happen last or else xarray will drop the dims added
    add_grid_size_dimensions(weight_fn, rof_scrip, ocn_file)

    # if we generated tmp scrip file(s) delete to avoid clutter
    if re.search("^/tmp/tmp", str(rof_scrip)):
        shutil.rmtree(rof_scrip.parent)


if __name__ == "__main__":
    gen_mapping_rof_to_ocn()
