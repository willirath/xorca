"""Test the pre-processing lib."""

from itertools import product
import numpy as np
import pytest
import xarray as xr

from xorca.lib import (create_minimal_coords_ds, get_name_dict, rename_dims,
                       trim_and_squeeze)
from xorca import orca_names


@pytest.mark.parametrize(
    "model_config_and_trimming",
    [{"model_config": "GLOBAL", "y": (1, -1), "x": (1, -1)},
     {"model_config": "NEST", "y": (None, None), "x": (None, None)},
     {"model_config": None, "y": (None, None), "x": (None, None)},
     {"model_config": "gLOBaL", "y": (1, -1), "x": (1, -1)},
     {"model_config": "nest", "y": (None, None), "x": (None, None)},
     {"model_config": None, "y": (None, None), "x": (None, None)}])
def test_trim_and_sqeeze_by_model_config(model_config_and_trimming):
    N = 102
    ds = xr.Dataset(
        coords={"degen": (["degen"], [1]),
                "t": (["t"], [1]),
                "y": (["y", ], range(N)),
                "x": (["x", ],  range(N))})

    model_config = model_config_and_trimming["model_config"]
    x_slice = model_config_and_trimming["x"]
    y_slice = model_config_and_trimming["y"]
    ds_t = trim_and_squeeze(ds, model_config=model_config)
    ds_trimmed_here = ds.isel(x=slice(*x_slice), y=slice(*y_slice))

    assert "degen" not in ds_t.dims
    assert ds_t.dims["y"] == ds_trimmed_here.dims["y"]
    assert ds_t.dims["x"] == ds_trimmed_here.dims["x"]


@pytest.mark.parametrize("y_slice", [(1, -1), (2, -2), None])
@pytest.mark.parametrize("x_slice", [(1, -1), (2, -2), None])
def test_trim_and_sqeeze_by_yx_slice(y_slice, x_slice):
    N = 102
    ds = xr.Dataset(
        coords={"degen": (["degen"], [1]),
                "y": (["y", ], range(N)),
                "x": (["x", ],  range(N))})

    ds_t = trim_and_squeeze(ds, y_slice=y_slice, x_slice=x_slice)

    # To also cover partial overrides, we check for None's here
    ds_trimmed_here = ds
    if y_slice is None:
        ds_trimmed_here = ds_trimmed_here.isel(y=slice(1, -1))
    else:
        ds_trimmed_here = ds_trimmed_here.isel(y=slice(*y_slice))
    if x_slice is None:
        ds_trimmed_here = ds_trimmed_here.isel(x=slice(1, -1))
    else:
        ds_trimmed_here = ds_trimmed_here.isel(x=slice(*x_slice))

    assert "degen" not in ds_t.dims
    assert ds_t.dims["y"] == ds_trimmed_here.dims["y"]
    assert ds_t.dims["x"] == ds_trimmed_here.dims["x"]


def test_create_minimal_coords_ds():
    Nz = 46
    Ny = 102
    Nx = 102

    source_ds = xr.Dataset(
        coords={
            "z": (["z", ], range(Nz)),
            "y": (["y", ], range(Ny)),
            "x": (["x", ], range(Nx))
        }
    )

    # This is THE central info:  This ``coords`` dictionary defines the whole
    # grid geometry with the U, V, and F grid points being shifted to the right
    # (that is further away from the origin of the dimension) and the W grid
    # points shifted to the left (that is closer to the origin of the
    # dimension).
    target_ds = xr.Dataset(
        coords={
            "z_c": (["z_c", ], np.arange(1, Nz + 1),
                    {"axis": "Z"}),
            "z_l": (["z_l", ], np.arange(1, Nz + 1) - 0.5,
                    {"axis": "Z", "c_grid_axis_shift": - 0.5}),
            "y_c": (["y_c", ], np.arange(1, Ny + 1),
                    {"axis": "Y"}),
            "y_r": (["y_r", ], np.arange(1, Ny + 1) + 0.5,
                    {"axis": "Y", "c_grid_axis_shift": 0.5}),
            "x_c": (["x_c", ], np.arange(1, Nx + 1),
                    {"axis": "X"}),
            "x_r": (["x_r", ], np.arange(1, Nx + 1) + 0.5,
                    {"axis": "X", "c_grid_axis_shift": 0.5})
        }
    )

    test_ds = create_minimal_coords_ds(source_ds)

    # Check presence and length of dimensions
    assert all([target_ds.dims[k] == test_ds.dims[k]
                for k in test_ds.dims.keys()])

    # Check coordinate values
    assert all([all(target_ds.coords[k] == test_ds.coords[k])
                for k in test_ds.coords.keys()])


_dims = {
    "t": ("t", "time_counter"),
    "z": ("z", "Z"),
    "y": ("y", "Y"),
    "x": ("x", "X")
}


@pytest.mark.parametrize(
    'dims',
    ({vk[1][ii]: vk[0]
      for vk, ii in zip(_dims.items(), i)}
     for i in product(range(2), repeat=4)))
def test_rename_dims(dims):
    source_dims = tuple(dims.keys())
    da = xr.DataArray(np.empty((0, 0, 0, 0)), dims=source_dims)
    da_renamed = rename_dims(da)

    assert all(d in da_renamed.dims for d in _dims.keys())
    assert all(v[1] not in da_renamed.dims for k, v in _dims.items())


@pytest.mark.parametrize("dict_name",
                         ["rename_dims", "orca_variables",
                          "orca_coords", "this_one_does_not_exist"])
@pytest.mark.parametrize("update_dict",
                         [{},
                          {"SIGMA": "sigma"},
                          {"my_tracer": {"dims": ["t", "z_c", "y_c", "x_c"]}}])
def test_get_name_dict(dict_name, update_dict):
    _kwargs = {"update_" + dict_name: update_dict}
    dict_updated = get_name_dict(dict_name, **_kwargs)

    dict_updated_here = orca_names.__dict__.get(dict_name, {}).copy()
    dict_updated_here.update(update_dict)

    assert all(
        (k in dict_updated_here.keys()) and
        (v == dict_updated_here[k])
        for k, v in dict_updated.items())

    assert all(
        (k in dict_updated.keys()) and
        (v == dict_updated[k])
        for k, v in dict_updated_here.items())
