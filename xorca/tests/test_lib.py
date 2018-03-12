"""Test the pre-processing lib."""

import numpy as np
from pathlib import Path
import pytest
import xarray as xr

from xorca.lib import (copy_coords, create_minimal_coords_ds, trim_and_squeeze)


def test_trim_and_sqeeze():
    """Make sure to trim 2 slices and drop singletons."""
    N = 102
    ds = xr.Dataset(
        coords={"degen": (["degen"], [1]),
                "y": (["y", ], range(N)),
                "x": (["x", ],  range(N))})

    ds_t = trim_and_squeeze(ds)

    assert "degen" not in ds_t.dims
    assert ds_t.dims["y"] == (N - 2)
    assert ds_t.dims["x"] == (N - 2)


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


def _get_empty_mesh_mask_for_nn_msh_3(dims):

    # define vars
    _vars = {
        "tmask": ("t", "z", "y", "x"),
        "umask": ("t", "z", "y", "x"),
        "vmask": ("t", "z", "y", "x"),
        "fmask": ("t", "z", "y", "x"),
        "tmaskutil": ("t", "y", "x"),
        "umaskutil": ("t", "y", "x"),
        "vmaskutil": ("t", "y", "x"),
        "fmaskutil": ("t", "y", "x"),
        "glamt": ("t", "y", "x"),
        "glamu": ("t", "y", "x"),
        "glamv": ("t", "y", "x"),
        "glamf": ("t", "y", "x"),
        "gphit": ("t", "y", "x"),
        "gphiu": ("t", "y", "x"),
        "gphiv": ("t", "y", "x"),
        "gphif": ("t", "y", "x"),
        "e1t": ("t", "y", "x"),
        "e1u": ("t", "y", "x"),
        "e1v": ("t", "y", "x"),
        "e1f": ("t", "y", "x"),
        "e2t": ("t", "y", "x"),
        "e2u": ("t", "y", "x"),
        "e2v": ("t", "y", "x"),
        "e2f": ("t", "y", "x"),
        "ff": ("t", "y", "x"),
        "mbathy": ("t", "y", "x"),
        "misf": ("t", "y", "x"),
        "isfdraft": ("t", "y", "x"),
        "e3t_0": ("t", "z", "y", "x"),
        "e3u_0": ("t", "z", "y", "x"),
        "e3v_0": ("t", "z", "y", "x"),
        "e3w_0": ("t", "z", "y", "x"),
        "gdept_0": ("t", "z", "y", "x"),
        "gdepu": ("t", "z", "y", "x"),
        "gdepv": ("t", "z", "y", "x"),
        "gdepw_0": ("t", "z", "y", "x"),
        "gdept_1d": ("t", "z"),
        "gdepw_1d": ("t", "z"),
        "e3t_1d": ("t", "z"),
        "e3w_1d": ("t", "z")
    }

    # create three types of empty arrays
    empty = {}
    for _dims in [("t", "z", "y", "x"), ("t", "y", "x"), ("t", "z")]:
        empty[_dims] = np.full(tuple(dims[d] for d in _dims), np.nan)

    # create coords and variable dicts for xr.Dataset
    coords = {k: range(v) for k, v in dims.items() if k is not "t"}
    data_vars = {k: (v, empty[v]) for k, v in _vars.items()}

    return xr.Dataset(coords=coords, data_vars=data_vars)


@pytest.mark.parametrize(
    'dims', [
        {"t": 1, "z": 46, "y": 100, "x": 100},
        pytest.param({"t": 1, "z": 46, "y": 1021, "x": 1442},
                     marks=pytest.mark.xfail)
    ])
def test_copy_coords(dims):
    mock_up_mm = _get_empty_mesh_mask_for_nn_msh_3(dims)
    mock_up_mm = trim_and_squeeze(mock_up_mm).squeeze()

    return_ds = create_minimal_coords_ds(mock_up_mm)
    return_ds = copy_coords(return_ds, mock_up_mm)
