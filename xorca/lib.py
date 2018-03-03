"""Library for the conversion from NEMO output to XGCM data sets."""

import numpy as np
import xarray as xr

orca_names = {}

orca_variables = {
    "vozocrtx": {"coords": ["t", "z_c", "y_c", "x_r"]},
    "vomecrty": {"coords": ["t", "z_c", "y_r", "x_c"]}
}
orca_names.update(orca_variables)

orca_scale_factors = {
    "e1t": {"coords": ["y_c", "x_c"]},
    "e2t": {"coords": ["y_c", "x_c"]},
    "e3t": {"coords": ["z_c", "y_c", "x_c"]},
    "e1u": {"coords": ["y_c", "x_r"]},
    "e2u": {"coords": ["y_c", "x_r"]},
    "e3u": {"coords": ["z_c", "y_c", "x_r"]},
    "e1v": {"coords": ["y_r", "x_c"]},
    "e2v": {"coords": ["y_r", "x_c"]},
    "e3v": {"coords": ["z_c", "y_r", "x_c"]}
}
orca_names.update(orca_scale_factors)

orca_masks = {
    "tmask": {"coords": ["z_c", "y_c", "x_c"]},
    "umask": {"coords": ["z_c", "y_c", "x_r"]},
    "vmask": {"coords": ["z_c", "y_r", "x_c"]},
    "fmask": {"coords": ["z_c", "y_r", "x_r"]},
    "tmaskatl": {"coords": ["y_c", "x_c"]},
    "tmaskind": {"coords": ["y_c", "x_c"]},
    "tmaskpac": {"coords": ["y_c", "x_c"]}
}
orca_names.update(orca_masks)

orca_coords = {
    "depth_c": {"coords": ["z_c", ], "old_name": "gdept_0"},
    "depth_l": {"coords": ["z_l", ], "old_name": "gdepw_0"},
    "llat_cc": {"coords": ["y_c", "x_c"], "old_name": "gphit"},
    "llat_cr": {"coords": ["y_c", "x_r"], "old_name": "gphiu"},
    "llat_rc": {"coords": ["y_r", "x_c"], "old_name": "gphiv"},
    "llat_rr": {"coords": ["y_r", "x_r"], "old_name": "gphif"},
    "llon_cc": {"coords": ["y_c", "x_c"], "old_name": "glamt"},
    "llon_cr": {"coords": ["y_c", "x_r"], "old_name": "glamu"},
    "llon_rc": {"coords": ["y_r", "x_c"], "old_name": "glamv"},
    "llon_rr": {"coords": ["y_r", "x_r"], "old_name": "glamf"}
}
orca_names.update(orca_coords)


def _rename_orca_names(ds):
    rename_dict = {v["old_name"]: k
                   for k, v in orca_names.items()
                   if "old_name" in v}
    return ds.rename(rename_dict)


def preprocess_orca(mm_file, ds):

    # First, construct data set from mesh-mask file
    ds_mm = xr.open_dataset(mm_file)
    ds_mm = ds_mm.isel(x=slice(1, -1), y=slice(1, -1))
    ds_mm = ds_mm.squeeze()

    N_z = len(ds_mm.coords["z"])
    N_y = len(ds_mm.coords["y"])
    N_x = len(ds_mm.coords["x"])

    ds_mm = _rename_orca_names(ds_mm)

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

    coords.update({k: (v["coords"], ds_mm[k].data)
                   for k, v in orca_coords.items()})

    return_ds = xr.Dataset(coords=coords)

    # Now, get the ds to be pre-processed
    ds = ds.isel(y=slice(1, -1), x=slice(1, -1))
    ds = ds.squeeze()

    # transfer time axis?
    try:
        ds = ds.rename({"time_counter": "t"})
        return_ds.coords["t"] = ds.coords["t"]
    except Exception:
        pass

    for var_name, names in orca_names.items():
        old_name = names.get("old_name", var_name)
        if old_name in ds:
            return_ds[var_name] = (names["coords"], ds[old_name].data)

    return return_ds
