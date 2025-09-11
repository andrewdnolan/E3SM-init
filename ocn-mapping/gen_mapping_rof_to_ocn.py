#!/usr/bin/env python3

import click
import logging
import numpy as np
import re
import shutil
import tempfile
import xarray as xr

from datetime import datetime
from logging import Logger
from mache import MachineInfo, discover_machine
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from pathlib import Path
from typing import Literal, NoReturn
from xarray.core.dataarray import DataArray

click_input_path = click.Path(
    exists=True, dir_okay=False, readable=True, path_type=Path
)

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
    parallel_executable: str | None = None,
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
    Convert input file, either a domain or rtm data file, to the SCRIP format.

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

    grid_dims = np.array((rof_ds.sizes["nj"], rof_ds.sizes["ni"]))

    rof_ds = rof_ds.stack(grid_size=("nj", "ni"), create_index=False)
    rof_ds = rof_ds.rename(nv="grid_corners")

    # rtm data files have "nt" dimension
    if "nt" in rof_ds.dims:
        rof_ds = rof_ds.drop_dims("nt")

    # ensure the dimensions are order as needed by the SCRIP format
    rof_ds = rof_ds.transpose("grid_size", "grid_corners")

    if ('xc' in rof_ds.coords) and ('yc' in rof_ds.coords):
        rof_ds = rof_ds.reset_coords(("xc", "yc"))

    # should the dataset also include area and/or imask variables
    scrip_ds = xr.Dataset({
        "grid_dims": xr.DataArray(grid_dims, dims="grid_rank"),
        "grid_center_lat": fmt_attrs(rof_ds["yc"]),
        "grid_center_lon": fmt_attrs(rof_ds["xc"]),
        "grid_corner_lat": fmt_attrs(rof_ds["yv"]),
        "grid_corner_lon": fmt_attrs(rof_ds["xv"]),
    })

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
    Mask the rof scrip file based on the rof mesh variable requested and
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

def validate_input_file(
        fp: Path, logger: Logger, limit_to: None | Literal["scrip"] = None,
) -> Path:
    """
    Validates input file's type and converts to SCRIP format if appropriate.

    Parameters
    ----------
    fp: pathlib.Path
        Path to the input file
    logger: logging.Logger
        Logger to stream stdout and stderr too
    limit_to: None or "scrip"
        Allows certain input files to have to be "scrip" files

    Returns
    -------
    fp: pathlib.Path
        Path to the validate input file OR the path to a temproary file, which
        is the input file converted to SCRIP format
    """

    def get_file_type(fp: Path) -> Literal["rtm", "domain", "scrip"]:
        """
        Get the type (rtm, domain, or scrip) of the input file.

        NOTE: Currently this soely does this based off the filename.
        It does not actually check that the infered file type is properly
        formatted, by checking dimensions and/or attributes.

        Parameters
        ----------
        fp: pathlib.Path
            Path to the input file

        Returns
        -------
        file_type: Literal["rtm", "domain", "scrip"]
            String describing the input files type
        """

        if "scrip" in str(fp.stem).lower():
            return "scrip"
        elif "domain" in str(fp.stem).lower():
            return "domain"
        elif "daitren" in str(fp.stem).lower():
            return "rtm"
        else:
            raise ValueError()

    file_type = get_file_type(fp)

    if limit_to == "scrip" and file_type != "scrip":
        raise ValueError()

    if file_type != "scrip":
        logger.info(
            f"Input file: {fp} is a {file_type} file. "
            f"Converting to SCRIP format."
        )

        fp = convert_to_scrip(fp)

    return fp

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
@click.option('--rof_mesh', type=Path)
@click.option('--mask_var', type=str)
@click.option('--lnd_domain', type=Path)
def gen_mapping_rof_to_ocn(
    rof_file, ocn_file, weight_fn, nprocs, rof_mesh, mask_var, lnd_domain
):
    """
    Generate an area weighted nearest neighbor mapping file
    """

    logger = create_logger()

    # accepts scrip, rtm, or domain files. If rtm or domain file is provided,
    # it is converted to scrip format and written to a tmp directory
    rof_scrip = validate_input_file(rof_file, logger)
    # b/c ocn mesh is unstructued, we only accept scrip files as inputs
    ocn_scrip = validate_input_file(ocn_file, logger, limit_to="scrip")

    if mask_var != None:

        if lnd_domain == None:
            raise ValueError()

        rof_scrip = mask_rof_scrip(rof_scrip, rof_mesh, mask_var, lnd_domain)

    parallel_executable = None
    if nprocs > 1:
        machine = discover_machine()
        config = MachineInfo(machine).config

        parallel_executable = config.get("parallel", "parallel_executable")

    generate_mapping_file(rof_scrip, ocn_scrip, weight_fn, logger,
                          parallel_executable=parallel_executable,
                          nprocs = nprocs)

    # if we generated tmp scrip file(s) delete to avoid clutter
    if re.search("^/tmp/tmp", str(rof_scrip)):
        shutil.rmtree(rof_scrip.parent)

    area_weight_mapping_file(weight_fn, logger)

if __name__ == "__main__":
    gen_mapping_rof_to_ocn()
