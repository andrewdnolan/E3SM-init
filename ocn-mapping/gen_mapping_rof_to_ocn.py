#!/usr/bin/env python3

import click
import logging
import numpy as np
import os
import re
import tempfile
import xarray as xr

from datetime import datetime
from mache import MachineInfo, discover_machine
from mpas_tools.io import write_netcdf
from mpas_tools.logging import check_call
from pathlib import Path

click_input_path = click.Path(
    exists=True, dir_okay=False, readable=True, path_type=Path
)

def calculate_spherical_quad_area(xv, yv):
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

def generate_mapping_file(rof_scrip, ocn_scrip, weight_fn, logger,
                          parallel_executable=None, nprocs=1):
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

    parallel_executable : str
        executable needed to launch a parallel job

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

def area_weight_mapping_file(weight_fn, logger):
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

def mask_rof_scrip(rof_scrip, rof_mesh, mask_var, lnd_domain):
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

    # create the filepath to a temporary scrip file
    tmp_dir = tempfile.mkdtemp()
    tmp_fn = rof_scrip.stem + f".CUSTOM_{mask_var}_MASK.nc"
    tmp_rof_scrip = Path(tmp_dir) / tmp_fn

    write_netcdf(scrip_ds, tmp_rof_scrip)

    return tmp_rof_scrip

def create_logger():
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
@click.option('-s', '--rof_scrip', type=click_input_path)
@click.option('-d', '--ocn_scrip', type=click_input_path)
@click.option('-w', '--weight_fn', type=Path)
@click.option('-n', '--nprocs', type=int, default=1)
@click.option('--rof_mesh', type=Path)
@click.option('--mask_var', type=str)
@click.option('--lnd_domain', type=Path)
def gen_mapping_rof_to_ocn(
    rof_scrip, ocn_scrip, weight_fn, nprocs, rof_mesh, mask_var, lnd_domain
):
    """
    Generate an area weighted nearest neighbor mapping file
    """

    logger = create_logger()

    if mask_var != None:
        rof_scrip = mask_rof_scrip(rof_scrip, rof_mesh, mask_var, lnd_domain)

    parallel_executable = None
    if nprocs > 1:
        machine = discover_machine()
        config = MachineInfo(machine).config

        parallel_executable = config.get("parallel", "parallel_executable")

    generate_mapping_file(rof_scrip, ocn_scrip, weight_fn, logger,
                          parallel_executable=parallel_executable,
                          nprocs = nprocs)

    # if we generated a tmp scrip file delete to avoid clutter
    if re.search("^/tmp/tmp", str(rof_scrip)):
        os.remove(rof_scrip)
        os.rmdir(rof_scrip.parent)

    area_weight_mapping_file(weight_fn, logger)

if __name__ == "__main__":
    gen_mapping_rof_to_ocn()
