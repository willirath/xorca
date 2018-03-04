"""Library for the conversion from NEMO output to XGCM data sets."""

import numpy as np
import xarray as xr

from . import orca_names


def trim_and_squeeze(ds):
    if "y" in ds:
        ds = ds.isel(y=slice(1, -1))
    if "x" in ds:
        ds = ds.isel(x=slice(1, -1))
    ds = ds.squeeze()
    return ds


def create_minimal_coords_ds(ds_mm):
    N_z = len(ds_mm.coords["z"])
    N_y = len(ds_mm.coords["y"])
    N_x = len(ds_mm.coords["x"])

    coords = {
        "z_c": (["z_c", ], np.arange(1, N_z + 1),
                {"axis": "Z"}),
        "z_l": (["z_l", ], np.arange(1, N_z + 1) - 0.5,
                {"axis": "Z", "c_grid_axis_shift": - 0.5}),
        "y_c": (["y_c", ], np.arange(1, N_y + 1),
                {"axis": "Y"}),
        "y_r": (["y_r", ], np.arange(1, N_y + 1) + 0.5,
                {"axis": "Y", "c_grid_axis_shift": 0.5}),
        "x_c": (["x_c", ], np.arange(1, N_x + 1),
                {"axis": "X"}),
        "x_r": (["x_r", ], np.arange(1, N_x + 1) + 0.5,
                {"axis": "X", "c_grid_axis_shift": 0.5})
    }

    return xr.Dataset(coords=coords)


def copy_coords(return_ds, ds_mm):
    for key, names in orca_names.orca_coords.items():
        new_name = key
        new_dims = names["dims"]
        old_name = names.get("old_name", new_name)
        if old_name in ds_mm.coords:
            return_ds.coords[new_name] = (new_dims,
                                          ds_mm.coords[old_name].data)
        if old_name in ds_mm:
            return_ds.coords[new_name] = (new_dims,
                                          ds_mm[old_name].data)
    return return_ds


def copy_vars(return_ds, raw_ds):
    for key, names in orca_names.orca_variables.items():
        new_name = key
        new_dims = names["dims"]
        old_name = names.get("old_name", new_name)
        if old_name in raw_ds:
            return_ds[new_name] = (new_dims, raw_ds[old_name].data)
    return return_ds


def rename_dims(ds):
    rename_dict = {
        k: v for k, v in orca_names.rename_dims.items()
        if k in ds.dims
    }
    return ds.rename(rename_dict)


def make_depth_positive_upward(ds):
    for k, v in orca_names.orca_coords.items():
        force_sign = v.get("force_sign", False)
        if force_sign and k in ds.coords:
            ds[k] = force_sign * abs(ds[k])

    return ds


def preprocess_orca(mm_file, ds):

    # construct minimal grid-aware data set from mesh-mask file
    ds_mm = xr.open_dataset(mm_file)
    ds_mm = trim_and_squeeze(ds_mm)
    return_ds = create_minimal_coords_ds(ds_mm)

    # make sure dims are called correctly and trim input ds
    ds = rename_dims(ds)
    ds = trim_and_squeeze(ds)

    # copy coordinates from the mesh-mask and from the data set
    return_ds = copy_coords(return_ds, ds_mm)
    return_ds = copy_coords(return_ds, ds)

    # copy variables from the data set
    return_ds = copy_vars(return_ds, ds)

    # Finally, make sure depth is positive upward
    return_ds = make_depth_positive_upward(return_ds)

    return return_ds
