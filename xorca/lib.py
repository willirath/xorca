"""Library for the conversion from NEMO output to XGCM data sets."""

from itertools import chain
import numpy as np
import xarray as xr

from . import orca_names


def trim_and_squeeze(ds,
                     model_config="GLOBAL",
                     y_slice=None, x_slice=None,
                     **kwargs):
    """Remove redundant grid points and drop singleton dimensions.

    Parameters
    ----------
    ds : xr Dataset | DataArray
        The object to trim.
    model_config : immutable
        Selects pre-defined trimming setup.  If omitted, or if the model_config
        is not known here, no trimming will be done.

        Possible configurations:
             - `"GLOBAL"` (*default*) : `.isel(y=slice(1, 11), x=slice(1, -1))`
             - `"NEST"` : No trimming
    y_slice : tuple
        How to slice in y-dimension?  `y_slice=(1, -1)` will slice from 1 to
        -1, which amounts to dropping the first and last index along the
        y-dimension.  This will override selection along y given by
        `model_config`.
    x_slice : tuple
        See y_slice.  This will override selection along x given by
        `model_config`.

    Returns
    -------
    trimmed ds

    """

    # Be case-insensitive
    if isinstance(model_config, str):
        model_config = model_config.upper()

    how_to_trim = {
        "GLOBAL": {"y": (1, -1), "x": (1, -1)},
        "NEST": {},
    }

    yx_slice_dict = how_to_trim.get(
        model_config, {})
    if y_slice is None:
        y_slice = yx_slice_dict.get("y")
    if x_slice is None:
        x_slice = yx_slice_dict.get("x")

    if (y_slice is not None) and ("y" in ds.dims):
        ds = ds.isel(y=slice(*y_slice))
    if (x_slice is not None) and ("x" in ds.dims):
        ds = ds.isel(x=slice(*x_slice))
    ds = ds.squeeze()
    return ds


def create_minimal_coords_ds(mesh_mask, **kwargs):
    """Create a minimal set of coordinates from a mesh-mask dataset.

    This creates `"central"` and `"right"` grid points for the horizontal grid
    and `"central"` and `"left"` grid points in the vertical.

    """
    N_z = len(mesh_mask.coords["z"])
    N_y = len(mesh_mask.coords["y"])
    N_x = len(mesh_mask.coords["x"])

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


def get_name_dict(dict_name, **kwargs):
    """Return potentially updated name dictionary.

    Parameters
    ----------
    dict_name : str
        Name of the dict from `xorca.orca_names` to be returned / updated.
        `get_name_dict` will look for a `kwarg` called `"update_" + dict_name`
        that will be used to override / add keys from `dict_name`.

        If `dict_name` is not in `xorca.orca_names`, an empty dict will be
        updated with `kwargs["update_" + dict_name]`.

    Returns
    -------
    dict
        Updated dict.


    Examples
    --------
    ```python
    print(get_name_dict("rename_dims"))
    # -> {"time_counter": "t", "Z": "z", "Y": "y", "X": "x"}

    print(get_name_dict("rename_dims", update_rename_dims={"SIGMA": "sigma"}))
    # -> {"time_counter": "t", "Z": "z", "Y": "y", "X": "x", "SIGMA": "sigma"}

    print(get_name_dict("not_defined", update_rename_dims={"SIGMA": "sigma"}))
    # -> {"SIGMA": "sigma"}
    ```
    """
    orig_dict = orca_names.__dict__.get(dict_name, {}).copy()
    update_dict = kwargs.get("update_" + dict_name, {})
    orig_dict.update(update_dict)
    return orig_dict


def copy_coords(return_ds, ds_in, **kwargs):
    """Copy coordinates and map them to the correct grid.

    This copies all coordinates defined in `xorca.orca_names.orca_coords` from
    `ds_in` to `return_ds`.
    """
    for key, names in get_name_dict("orca_coords", **kwargs).items():
        new_name = key
        new_dims = names["dims"]
        for old_name in names.get("old_names", [new_name, ]):

            # This will first try and copy `old_name` from the input ds coords
            # and then from the input ds variables.  As soon as a ds can be
            # copied sucessfully (that is , if they are present and have the
            # correct shape), the loop is broken and the next target coordinate
            # will be built.
            if old_name in ds_in.coords:
                try:
                    return_ds.coords[new_name] = (new_dims,
                                                  ds_in.coords[old_name].data)
                    break
                except ValueError as e:
                    pass
            if old_name in ds_in:
                try:
                    return_ds.coords[new_name] = (new_dims,
                                                  ds_in[old_name].data)
                    break
                except ValueError as e:
                    pass

    return return_ds


def copy_vars(return_ds, raw_ds, **kwargs):
    """Copy variables and map them to the correct grid.

    This copies all variables defined in `xorca.orca_names.orca_variables` from
    `raw_ds` to `return_ds`.
    """
    for key, names in get_name_dict("orca_variables", **kwargs).items():
        new_name = key
        new_dims = names["dims"]
        old_names = names.get("old_names", [new_name, ])
        for old_name in old_names:
            if old_name in raw_ds:
                try:
                    return_ds[new_name] = (new_dims, raw_ds[old_name].data)
                    break
                except ValueError as e:
                    pass
    return return_ds


def rename_dims(ds, **kwargs):
    """Rename dimensions.

    This renames all dimensions defined in `xorca.orca_names.rename_dims` and
    returns the data set with renamed dimensinos.
    """
    rename_dict = {
        k: v for k, v in get_name_dict("rename_dims", **kwargs).items()
        if k in ds.dims
    }
    return ds.rename(rename_dict)


def force_sign_of_coordinate(ds, **kwargs):
    """Force definite sign of coordinates.

    For all coordinates defined in `xorca.orca_names.orca_coordinates`, enforce
    a sign if there is an item telling us to do so.  This is most useful to
    ensure that, e.g., depth is _always_ pointing upwards or downwards.
    """
    for k, v in get_name_dict("orca_coords", **kwargs).items():
        force_sign = v.get("force_sign", False)
        if force_sign and k in ds.coords:
            ds[k] = force_sign * abs(ds[k])

    return ds


def open_mf_or_dataset(data_files, **kwargs):
    """Open data_files as multi-file or a single-file xarray Dataset."""

    try:
        mesh_mask = xr.open_mfdataset(data_files, chunks={})
    except TypeError as e:
        mesh_mask = xr.open_dataset(data_files, chunks={})

    return mesh_mask


def get_all_compatible_chunk_sizes(chunks, dobj):
    """Return only thos chunks that are compatible with the given data.

    Parameters
    ----------
    chunks : dict
        Dictionary with all possible chunk sizes.  (Keys are dimension names,
        values are integers for the corresponding chunk size.)
    dobj : dataset or data array
        Dimensions of dobj will be used to filter the `chunks` dict.

    Returns
    -------
    dict
        Dictionary with only those items of `chunks` that can be applied to
        `dobj`.
    """
    return {k: v for k, v in chunks.items() if k in dobj.dims}


def set_time_independent_vars_to_coords(ds):
    """Make sure all time-independent variables are coordinates."""
    return ds.set_coords([v for v in ds.data_vars.keys()
                          if 't' not in ds[v].dims],
                         inplace=False)


def preprocess_orca(mesh_mask, ds, **kwargs):
    """Preprocess orca datasets before concatenating.

    This is meant to be used like:
    ```python
    ds = xr.open_mfdataset(
        data_files,
        preprocess=(lambda ds:
                    preprocess_orca(mesh_mask, ds)))
    ```

    Parameters
    ----------
    mesh_mask : Dataset | Path | sequence | string
        An xarray `Dataset` or anything accepted by `xr.open_mfdataset` or,
        `xr.open_dataset`: A single file name, a sequence of Paths or file
        names, a glob statement.
    ds : xarray dataset
        Xarray dataset to be processed before concatenating.
    input_ds_chunks : dict
        Chunks for the ds to be preprocessed.  Pass chunking for any input
        dimension that might be in the input data.

    Returns
    -------
    xarray dataset

    """
    # make sure input ds is chunked
    input_ds_chunks = get_all_compatible_chunk_sizes(
        kwargs.get("input_ds_chunks", {}), ds)
    ds = ds.chunk(input_ds_chunks)

    # construct minimal grid-aware data set from mesh-mask info
    if not isinstance(mesh_mask, xr.Dataset):
        mesh_mask = open_mf_or_dataset(mesh_mask, **kwargs)
    mesh_mask = trim_and_squeeze(mesh_mask, **kwargs)
    return_ds = create_minimal_coords_ds(mesh_mask, **kwargs)

    # make sure dims are called correctly and trim input ds
    ds = rename_dims(ds, **kwargs)
    ds = trim_and_squeeze(ds, **kwargs)

    # copy coordinates from the mesh-mask and from the data set
    return_ds = copy_coords(return_ds, mesh_mask, **kwargs)
    return_ds = copy_coords(return_ds, ds, **kwargs)

    # copy variables from the data set
    return_ds = copy_vars(return_ds, ds, **kwargs)

    # make sure depth is positive upward
    return_ds = force_sign_of_coordinate(return_ds, **kwargs)

    # make everything that does not depend on time a coord
    return_ds = set_time_independent_vars_to_coords(return_ds)

    return return_ds


def load_xorca_dataset(data_files=None, aux_files=None, decode_cf=True,
                       **kwargs):
    """Create a grid-aware NEMO dataset.

    Parameters
    ----------
    data_files : Path | sequence | string
        Anything accepted by `xr.open_mfdataset` or, `xr.open_dataset`: A
        single file name, a sequence of Paths or file names, a glob statement.
    aux_files : Path | sequence | string
        Anything accepted by `xr.open_mfdataset` or, `xr.open_dataset`: A
        single file name, a sequence of Paths or file names, a glob statement.
    input_ds_chunks : dict
        Chunks for the ds to be preprocessed.  Pass chunking for any input
        dimension that might be in the input data.
    target_ds_chunks : dict
        Chunks for the final data set.  Pass chunking for any of the likely
        output dims: `("t", "z_c", "z_l", "y_c", "y_r", "x_c", "x_r")`
    decode_cf : bool
        Do we want the CF decoding to be done already?  Default is True.

    Returns
    -------
    dataset

    """

    default_input_ds_chunks = {
        "time_counter": 1, "t": 1,
        "z": 2, "deptht": 2, "depthu": 2, "depthv": 2, "depthw": 2,
        "y": 200, "x": 200
    }
    input_ds_chunks = kwargs.get("input_ds_chunks",
                                 default_input_ds_chunks)

    default_target_ds_chunks = {
        "t": 1,
        "z_c": 2, "z_l": 2,
        "y_c": 200, "y_r": 200,
        "x_c": 200, "x_r": 200
    }
    target_ds_chunks = kwargs.get("target_ds_chunks",
                                  default_target_ds_chunks)

    # First, read aux files to learn about all dimensions.  Then, open again
    # and specify chunking for all applicable dims.  It is very important to
    # already pass the `chunks` arg to `open_[mf]dataset`, to ensure
    # distributed performance.
    with xr.open_mfdataset(aux_files, decode_cf=decode_cf) as _aux_ds:
        aux_ds_chunks = get_all_compatible_chunk_sizes(
            input_ds_chunks, _aux_ds)
    aux_ds = xr.open_mfdataset(aux_files, chunks=aux_ds_chunks,
                               decode_cf=decode_cf)

    # Again, we first have to open all data sets to filter the input chunks.
    _data_files_chunks = map(
        lambda df: get_all_compatible_chunk_sizes(
            input_ds_chunks, xr.open_dataset(df, decode_cf=decode_cf)),
        data_files)
    ds_xorca = xr.merge(
        map(
            lambda ds: preprocess_orca(aux_ds, ds),
            chain(
                map(lambda df, chunks: xr.open_dataset(df, chunks=chunks,
                                                       decode_cf=decode_cf),
                    data_files, _data_files_chunks),
                [aux_ds, ])))

    # Chunk the final ds
    ds_xorca = ds_xorca.chunk(
        get_all_compatible_chunk_sizes(target_ds_chunks, ds_xorca))

    return ds_xorca
