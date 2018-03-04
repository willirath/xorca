"""Test the pre-processing lib."""

import numpy as np
import xarray as xr

from xorca.lib import create_minimal_coords_ds, trim_and_squeeze


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
