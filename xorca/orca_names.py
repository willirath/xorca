"""Collect names, grids, etc."""

orca_coords = {
    "t": {"dims": ["t", ]},
    "depth_c": {"dims": ["z_c", ], "old_name": "gdept_0", "force_sign": -1.0},
    "depth_l": {"dims": ["z_l", ], "old_name": "gdepw_0", "force_sign": -1.0},
    "llat_cc": {"dims": ["y_c", "x_c"], "old_name": "gphit"},
    "llat_cr": {"dims": ["y_c", "x_r"], "old_name": "gphiu"},
    "llat_rc": {"dims": ["y_r", "x_c"], "old_name": "gphiv"},
    "llat_rr": {"dims": ["y_r", "x_r"], "old_name": "gphif"},
    "llon_cc": {"dims": ["y_c", "x_c"], "old_name": "glamt"},
    "llon_cr": {"dims": ["y_c", "x_r"], "old_name": "glamu"},
    "llon_rc": {"dims": ["y_r", "x_c"], "old_name": "glamv"},
    "llon_rr": {"dims": ["y_r", "x_r"], "old_name": "glamf"}
}

orca_variables = {
    "vozocrtx": {"dims": ["t", "z_c", "y_c", "x_r"]},
    "vomecrty": {"dims": ["t", "z_c", "y_r", "x_c"]},
    "e1t": {"dims": ["y_c", "x_c"]},
    "e2t": {"dims": ["y_c", "x_c"]},
    "e3t": {"dims": ["z_c", "y_c", "x_c"]},
    "e1u": {"dims": ["y_c", "x_r"]},
    "e2u": {"dims": ["y_c", "x_r"]},
    "e3u": {"dims": ["z_c", "y_c", "x_r"]},
    "e1v": {"dims": ["y_r", "x_c"]},
    "e2v": {"dims": ["y_r", "x_c"]},
    "e3v": {"dims": ["z_c", "y_r", "x_c"]},
    "tmask": {"dims": ["z_c", "y_c", "x_c"]},
    "umask": {"dims": ["z_c", "y_c", "x_r"]},
    "vmask": {"dims": ["z_c", "y_r", "x_c"]},
    "fmask": {"dims": ["z_c", "y_r", "x_r"]},
    "tmaskatl": {"dims": ["y_c", "x_c"]},
    "tmaskind": {"dims": ["y_c", "x_c"]},
    "tmaskpac": {"dims": ["y_c", "x_c"]},
    "umaskatl": {"dims": ["y_c", "x_r"], "old_name": "tmaskatl"},
    "umaskind": {"dims": ["y_c", "x_r"], "old_name": "tmaskind"},
    "umaskpac": {"dims": ["y_c", "x_r"], "old_name": "tmaskpac"},
    "vmaskatl": {"dims": ["y_r", "x_c"], "old_name": "tmaskatl"},
    "vmaskind": {"dims": ["y_r", "x_c"], "old_name": "tmaskind"},
    "vmaskpac": {"dims": ["y_r", "x_c"], "old_name": "tmaskpac"},
    "fmaskatl": {"dims": ["y_r", "x_r"], "old_name": "tmaskatl"},
    "fmaskind": {"dims": ["y_r", "x_r"], "old_name": "tmaskind"},
    "fmaskpac": {"dims": ["y_r", "x_r"], "old_name": "tmaskpac"}
}

rename_dims = {
    "time_counter": "t",
    "Z": "z",
    "Y": "y",
    "X": "x"
}
