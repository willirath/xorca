"""Test grid autodetection."""

from xorca.lib import (render_depth_name, render_lat_name,
                       render_lon_name, arrays_are_close,
                       detect_horizontal_grid, is_depth_coord,
                       detect_vertical_grid, find_depth_coord_name)

import numpy as np
import pytest
import xarray as xr


@pytest.mark.parametrize("zgrid", ["c", "l"])
@pytest.mark.parametrize("ygrid", ["c", "r"])
@pytest.mark.parametrize("xgrid", ["c", "r"])
def test_render_coordname(zgrid, ygrid, xgrid):
    grid_dict = {"Z": zgrid, "Y": ygrid, "X": xgrid}
    assert ("depth_" + zgrid) == render_depth_name(grid_dict)
    assert ("lat_" + ygrid + xgrid) == render_lat_name(grid_dict)
    assert ("lon_" + ygrid + xgrid) == render_lon_name(grid_dict)


@pytest.mark.parametrize("size", [(10, 10), (100, 999)])
@pytest.mark.parametrize("atol", [1e-7, 1e-12, 0.0])
@pytest.mark.parametrize("rng_seed", range(3))
def test_array_comparison(size, atol, rng_seed):
    np.random.seed(rng_seed)
    ref_array = np.random.normal(size=size)
    disturbed_array = ref_array + atol * np.random.normal(size=size)
    assert arrays_are_close(disturbed_array, ref_array, atol=atol * 10)


@pytest.mark.parametrize("size", [(10, 10), (100, 999)])
@pytest.mark.parametrize("atol", [1e-7, 1e-12, 0.0])
@pytest.mark.parametrize("rng_seed", range(3))
def test_detection_of_horizontal_grid(size, atol, rng_seed):
    lat_vec = np.linspace(-90, 90, size[0])
    lon_vec = np.linspace(0, 360, size[1])

    lat_cc, lon_cc = np.meshgrid(lon_vec, lat_vec)

    dlat = np.diff(lat_vec)[0]
    dlon = np.diff(lon_vec)[0]

    lat_rc, lon_rc = lat_cc + dlat / 2, lon_cc + 0.0
    lat_cr, lon_cr = lat_cc + 0.0, lon_cc + dlon / 2
    lat_rr, lon_rr = lat_cc + dlon / 2, lon_cc + dlon / 2

    coords = {"lat_cc": xr.DataArray(lat_cc, dims=["y_c", "x_c"]),
              "lon_cc": xr.DataArray(lon_cc, dims=["y_c", "x_c"]),
              "lat_rc": xr.DataArray(lat_rc, dims=["y_r", "x_c"]),
              "lon_rc": xr.DataArray(lon_rc, dims=["y_r", "x_c"]),
              "lat_cr": xr.DataArray(lat_cr, dims=["y_c", "x_r"]),
              "lon_cr": xr.DataArray(lon_cr, dims=["y_c", "x_r"]),
              "lat_rr": xr.DataArray(lat_rr, dims=["y_r", "x_r"]),
              "lon_rr": xr.DataArray(lon_rr, dims=["y_r", "x_r"])}

    ds_coords = xr.Dataset(coords=coords)

    possible_grids = [{"Y": yg, "X": xg}
                      for xg in ["c", "r"] for yg in ["c", "r"]]

    np.random.seed(rng_seed)
    for pg in possible_grids:
        ds = xr.Dataset(coords={
            "nav_lat": (atol / 10 * np.random.uniform(size=size)
                        + coords[render_lat_name(pg)]),
            "nav_lon": (atol / 10 * np.random.uniform(size=size)
                        + coords[render_lon_name(pg)])
        })
        detected_grid = detect_horizontal_grid(ds, ds_coords)
        assert pg["Y"] == detected_grid["Y"]
        assert pg["X"] == detected_grid["X"]


def test_find_depth_coord():
    assert is_depth_coord("deptht")
    assert is_depth_coord("depthu")
    assert is_depth_coord("depthv")
    assert is_depth_coord("depthw")
    assert is_depth_coord("z")


def test_find_depth_coord():
    for cn in ["depth" + g for g in ["t", "u", "v", "w"]] + ["z", ]:
        ds = xr.Dataset(coords={cn: xr.DataArray([1, 2, 3])})
        assert find_depth_coord_name(ds) == cn


@pytest.mark.parametrize("size", [10, 999])
@pytest.mark.parametrize("atol", [1e-7, 1e-12, 0.0])
@pytest.mark.parametrize("rng_seed", range(3))
@pytest.mark.parametrize("depth_name", (["depth" + g
                                         for g in ["t", "u", "v", "w"]]
                                        + ["z", ]))
def test_detection_of_vertical_grid(size, atol, rng_seed, depth_name):
        depth_vec = np.linspace(10000 / size, 10000, size)
        delta = np.diff(depth_vec)[0]

        depth_c = depth_vec
        depth_l = depth_vec - delta / 2

        coords = {"depth_c": xr.DataArray(depth_c, dims=["z_c", ]),
                  "depth_l": xr.DataArray(depth_l, dims=["z_l", ])}

        ds_coords = xr.Dataset(coords=coords)

        possible_grids = [{"Z": zg} for zg in ["c", "l"]]

        np.random.seed(rng_seed)
        for pg in possible_grids:
            ds = xr.Dataset(coords={
                depth_name: (atol / 10 * np.random.uniform(size=size)
                            + coords[render_depth_name(pg)])})
            detected_grid = detect_vertical_grid(ds, ds_coords)
            assert pg["Z"] == detected_grid["Z"]
