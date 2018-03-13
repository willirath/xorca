"""Test reading the mesh masks."""

import numpy as np
from pathlib import Path
import pytest
import xarray as xr

from xorca.lib import (copy_coords, copy_vars, create_minimal_coords_ds,
                       force_sign_of_coordinate, open_mf_or_dataset,
                       preprocess_orca, trim_and_squeeze)


# Seed the RNG
np.random.RandomState(seed=137)

# This is derived from
# `meshmask/ORCA025.L46.LIM2vp.JRA.XIOS2.KMS-T002_mask.nc`,
# `meshmask/ORCA025.L46.LIM2vp.JRA.XIOS2.KMS-T002_mesh_hgr.nc`, and
# `meshmask/ORCA025.L46.LIM2vp.JRA.XIOS2.KMS-T002_mesh_zgr.nc` in the example
# data:
_mm_vars_nn_msh_3 = {
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

# This is derived from `ORCA05.L46-KKG36F25H/mesh_mask.nc` in the example data:
_mm_vars_old = {
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
    "e3t": ("t", "z", "y", "x"),
    "e3u": ("t", "z", "y", "x"),
    "e3v": ("t", "z", "y", "x"),
    "e3w": ("t", "z", "y", "x"),
    "ff": ("t", "y", "x"),
    "mbathy": ("t", "y", "x"),
    "hdept": ("t", "y", "x"),
    "hdepw": ("t", "y", "x"),
    "e3t_ps": ("t", "y", "x"),
    "e3w_ps": ("t", "y", "x"),
    "gdept_0": ("t", "z"),
    "gdepw_0": ("t", "z"),
    "e3t_0": ("t", "z"),
    "e3w_0": ("t", "z")
}

# Same as `_mm_vars_nn_msh_3` but with additional variables
_mm_vars_nn_msh_3_added_misshaped_vars = _mm_vars_nn_msh_3
_mm_vars_nn_msh_3_added_misshaped_vars.update({
    "e3t": ("t", "z"),
    "e3u": ("t", "z"),
    "e3v": ("t", "z"),
    "e3w": ("t", "z")})

# Same as `_mm_vars_nn_msh_3` but with additional / modified variables
_mm_vars_old_added_misshaped_vars = _mm_vars_old
_mm_vars_old_added_misshaped_vars.update({
    "e3t": ("t", "z"),
    "e3u": ("t", "z"),
    "e3v": ("t", "z"),
    "e3w": ("t", "z"),
    "e3t_0": ("t", "z", "y", "x"),
    "e3u_0": ("t", "z", "y", "x"),
    "e3v_0": ("t", "z", "y", "x"),
    "e3w_0": ("t", "z", "y", "x"),
    "gdept_1d": ("t", "z", "y", "x"),
    "gdepw_1d": ("t", "z", "y", "x")})


def _get_nan_filled_data_set(dims, variables):

    # create three types of empty arrays
    empty = {}
    for _dims in [("t", "z", "y", "x"), ("t", "y", "x"), ("t", "z")]:
        empty[_dims] = np.full(tuple(dims[d] for d in _dims), np.nan)

    # create coords and variable dicts for xr.Dataset
    coords = {k: range(v) for k, v in dims.items() if k is not "t"}
    data_vars = {k: (v, empty[v]) for k, v in variables.items()}

    return xr.Dataset(coords=coords, data_vars=data_vars)


@pytest.mark.parametrize('set_mm_coords', [False, True])
@pytest.mark.parametrize('variables',
                         [_mm_vars_nn_msh_3,
                          _mm_vars_old,
                          _mm_vars_nn_msh_3_added_misshaped_vars])
@pytest.mark.parametrize(
    'dims', [
        {"t": 1, "z": 46, "y": 100, "x": 100},
        pytest.param({"t": 1, "z": 46, "y": 1021, "x": 1442},
                     marks=pytest.mark.xfail)
    ])
def test_copy_coords(dims, variables, set_mm_coords):
    mock_up_mm = _get_nan_filled_data_set(dims, variables)
    mock_up_mm = trim_and_squeeze(mock_up_mm).squeeze()

    if set_mm_coords:
        mock_up_mm = mock_up_mm.set_coords(
            [v for v in mock_up_mm.data_vars.keys()])

    return_ds = create_minimal_coords_ds(mock_up_mm)
    return_ds = copy_coords(return_ds, mock_up_mm)

    target_coords = [
        "depth_c", "depth_l",
        "llat_cc", "llat_cr", "llat_rc", "llat_rr",
        "llon_cc", "llon_cr", "llon_rc", "llon_rr"]

    assert all((tc in return_ds.coords) for tc in target_coords)


@pytest.mark.parametrize('variables',
                         [_mm_vars_nn_msh_3,
                          _mm_vars_nn_msh_3_added_misshaped_vars,
                          _mm_vars_old])
@pytest.mark.parametrize(
    'dims', [
        {"t": 1, "z": 46, "y": 100, "x": 100},
        pytest.param({"t": 1, "z": 46, "y": 1021, "x": 1442},
                     marks=pytest.mark.xfail)
    ])
def test_copy_vars(dims, variables):
    mock_up_mm = _get_nan_filled_data_set(dims, variables)
    mock_up_mm = trim_and_squeeze(mock_up_mm).squeeze()

    return_ds = create_minimal_coords_ds(mock_up_mm)
    return_ds = copy_vars(return_ds, mock_up_mm)

    target_vars = [
        "e1t", "e2t", "e3t", "e1u", "e2u", "e3u", "e1v", "e2v", "e3v",
        "tmask", "umask", "vmask", "fmask"]

    assert all((tv in return_ds.data_vars) for tv in target_vars)


@pytest.mark.parametrize('depth_c',
                         [np.random.randn(46),
                          np.abs(np.random.randn(46)),
                          np.abs(np.random.randn(46))])
@pytest.mark.parametrize('depth_l',
                         [np.random.randn(46),
                          np.abs(np.random.randn(46)),
                          np.abs(np.random.randn(46))])
def test_force_sign(depth_c, depth_l):
    input_ds = xr.Dataset({
        "depth_c": (["z_c", ], depth_c),
        "depth_l": (["z_l", ], depth_l),
    })
    input_ds = input_ds.set_coords(("depth_c", "depth_l"))
    return_ds = force_sign_of_coordinate(input_ds)

    assert all(return_ds.coords["depth_c"].data <= 0)
    assert all(return_ds.coords["depth_l"].data <= 0)


@pytest.mark.parametrize('use_pathlib', [False, True])
@pytest.mark.parametrize('use_sequence', [False, True])
@pytest.mark.parametrize('set_mm_coords', [False, True])
@pytest.mark.parametrize('variables',
                         [_mm_vars_nn_msh_3,
                          _mm_vars_old])
@pytest.mark.parametrize(
    'dims', [
        {"t": 1, "z": 46, "y": 100, "x": 100},
        {"t": 1, "z": 46, "y": 222, "x": 222},
    ])
def test_reading_mm_file(tmpdir, variables, dims, set_mm_coords,
                         use_sequence, use_pathlib):
    mock_up_mm = _get_nan_filled_data_set(dims, variables)
    if set_mm_coords:
        mock_up_mm = mock_up_mm.set_coords(
            [v for v in mock_up_mm.data_vars.keys()])

    file_name = str(tmpdir.join("mesh_mask.nc"))

    mock_up_mm.to_netcdf(file_name)

    if use_pathlib:
        file_name = Path(file_name)
    if use_sequence:
        file_name = [file_name, ]

    re_read_mm = open_mf_or_dataset(file_name)

    assert all(v in mock_up_mm for v in re_read_mm.data_vars)
    assert all(v in re_read_mm for v in mock_up_mm.data_vars)
    assert all(c in mock_up_mm.coords for c in re_read_mm.coords)
    assert all(c in re_read_mm.coords for c in mock_up_mm.coords)


@pytest.mark.parametrize('set_mm_coords', [False, True])
@pytest.mark.parametrize('variables',
                         [_mm_vars_nn_msh_3,
                          _mm_vars_old,
                          _mm_vars_nn_msh_3_added_misshaped_vars])
@pytest.mark.parametrize(
    'dims', [
        {"t": 1, "z": 46, "y": 100, "x": 100},
        {"t": 1, "z": 46, "y": 222, "x": 222},
    ])
def test_preprocess_orcea(tmpdir, variables, dims, set_mm_coords):
    mock_up_mm = _get_nan_filled_data_set(dims, variables)
    if set_mm_coords:
        mock_up_mm = mock_up_mm.set_coords(
            [v for v in mock_up_mm.data_vars.keys()])

    file_name = str(tmpdir.join("mesh_mask.nc"))

    mock_up_mm.to_netcdf(file_name)

    # TODO: Check that the returned dataset is actually looking good
    preprocess_orca(file_name, mock_up_mm)
