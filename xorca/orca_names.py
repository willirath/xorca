"""Collect names, grids, etc."""

# Note that as copying will loop over "old_names" und stop looking for
# condidates to copy as soon as one is found that fits.  This yields an
# implicit way of prioritizing variables.
orca_coords = {
    "t": {"dims": ["t", ]},
    "depth_c": {"dims": ["z_c", ],
                "old_names": ["gdept_1d", "gdept_0"],
                "force_sign": -1.0},
    "depth_l": {"dims": ["z_l", ],
                "old_names": ["gdepw_1d", "gdepw_0"],
                "force_sign": -1.0},
    "llat_cc": {"dims": ["y_c", "x_c"], "old_names": ["gphit", ]},
    "llat_cr": {"dims": ["y_c", "x_r"], "old_names": ["gphiu", ]},
    "llat_rc": {"dims": ["y_r", "x_c"], "old_names": ["gphiv", ]},
    "llat_rr": {"dims": ["y_r", "x_r"], "old_names": ["gphif", ]},
    "llon_cc": {"dims": ["y_c", "x_c"], "old_names": ["glamt", ]},
    "llon_cr": {"dims": ["y_c", "x_r"], "old_names": ["glamu", ]},
    "llon_rc": {"dims": ["y_r", "x_c"], "old_names": ["glamv", ]},
    "llon_rr": {"dims": ["y_r", "x_r"], "old_names": ["glamf", ]}
}

orca_variables = {
    "sobowlin": {"dims": ["t", "y_c", "x_c"]},
    "sohefldo": {"dims": ["t", "y_c", "x_c"]},
    "sohefldp": {"dims": ["t", "y_c", "x_c"]},
    "somixhgt": {"dims": ["t", "y_c", "x_c"]},
    "somxl010": {"dims": ["t", "y_c", "x_c"]},
    "sosaline": {"dims": ["t", "y_c", "x_c"]},
    "soshfldo": {"dims": ["t", "y_c", "x_c"]},
    "sossheig": {"dims": ["t", "y_c", "x_c"]},
    "sosstsst": {"dims": ["t", "y_c", "x_c"]},
    "sowafldp": {"dims": ["t", "y_c", "x_c"]},
    "sowaflup": {"dims": ["t", "y_c", "x_c"]},
    "sowindsp": {"dims": ["t", "y_c", "x_c"]},
    "vosaline": {"dims": ["t", "z_c", "y_c", "x_c"]},
    "votemper": {"dims": ["t", "z_c", "y_c", "x_c"]},
    "rhd": {"dims": ["t", "z_c", "y_c", "x_c"]},
    "sozotaux": {"dims": ["t", "y_c", "x_r"]},
    "vozocrtx": {"dims": ["t", "z_c", "y_c", "x_r"]},
    "uo": {"dims": ["t", "z_c", "y_c", "x_r"]},
    "vozoeivu": {"dims": ["t", "z_c", "y_c", "x_r"]},
    "sometauy": {"dims": ["t", "y_r", "x_c"]},
    "vomecrty": {"dims": ["t", "z_c", "y_r", "x_c"]},
    "vo": {"dims": ["t", "z_c", "y_r", "x_c"]},
    "vomeeivv": {"dims": ["t", "z_c", "y_r", "x_c"]},
    "ice_pres": {"dims": ["t", "y_c", "x_c"]},
    "iicenflx": {"dims": ["t", "y_c", "x_c"]},
    "iiceprod": {"dims": ["t", "y_c", "x_c"]},
    "iicesflx": {"dims": ["t", "y_c", "x_c"]},
    "iicestru": {"dims": ["t", "y_c", "x_c"]},
    "iicestrv": {"dims": ["t", "y_c", "x_c"]},
    "iicetemp": {"dims": ["t", "y_c", "x_c"]},
    "iicethic": {"dims": ["t", "y_c", "x_c"]},
    "iicevelu": {"dims": ["t", "y_c", "x_c"]},
    "iicevelv": {"dims": ["t", "y_c", "x_c"]},
    "ileadfra": {"dims": ["t", "y_c", "x_c"]},
    "ioceflxb": {"dims": ["t", "y_c", "x_c"]},
    "isnowpre": {"dims": ["t", "y_c", "x_c"]},
    "isnowthi": {"dims": ["t", "y_c", "x_c"]},
    "sobarstf": {"dims": ["t", "y_r", "x_r"]},
    "zomsfatl": {"dims": ["t", "z_l", "y_r"]},
    "zomsfglo": {"dims": ["t", "z_l", "y_r"]},
    "zomsfind": {"dims": ["t", "z_l", "y_r"]},
    "zomsfinp": {"dims": ["t", "z_l", "y_r"]},
    "zomsfpac": {"dims": ["t", "z_l", "y_r"]},
    "vovecrtz": {"dims": ["t", "z_l", "y_c", "x_c"]},
    "e1t": {"dims": ["y_c", "x_c"]},
    "e2t": {"dims": ["y_c", "x_c"]},
    # "e3t": {"dims": ["z_c", "y_c", "x_c"], "old_names": ["e3t", "e3t_0"]},
    "e3t": {"dims": ["t", "z_c", "y_c", "x_c"]},
    "e1u": {"dims": ["y_c", "x_r"]},
    "e2u": {"dims": ["y_c", "x_r"]},
    # "e3u": {"dims": ["z_c", "y_c", "x_r"], "old_names": ["e3u", "e3u_0"]},
    "e3u": {"dims": ["t", "z_c", "y_c", "x_r"]},
    "e1v": {"dims": ["y_r", "x_c"]},
    "e2v": {"dims": ["y_r", "x_c"]},
    # "e3v": {"dims": ["z_c", "y_r", "x_c"], "old_names": ["e3v", "e3v_0"]},
    "e3v": {"dims": ["t", "z_c", "y_r", "x_c"]},
    "e1f": {"dims": ["y_r", "x_r"]},
    "e2f": {"dims": ["y_r", "x_r"]},
    # "e3w": {"dims": ["z_l", "y_c", "x_c"], "old_names": ["e3w", "e3w_0"]},
    "e3w": {"dims": ["t", "z_l", "y_c", "x_c"]},
    "tmask": {"dims": ["z_c", "y_c", "x_c"]},
    "umask": {"dims": ["z_c", "y_c", "x_r"]},
    "vmask": {"dims": ["z_c", "y_r", "x_c"]},
    "fmask": {"dims": ["z_c", "y_r", "x_r"]},
    "tmaskatl": {"dims": ["y_c", "x_c"]},
    "tmaskind": {"dims": ["y_c", "x_c"]},
    "tmaskpac": {"dims": ["y_c", "x_c"]},
    "umaskatl": {"dims": ["y_c", "x_r"], "old_names": ["tmaskatl", ]},
    "umaskind": {"dims": ["y_c", "x_r"], "old_names": ["tmaskind", ]},
    "umaskpac": {"dims": ["y_c", "x_r"], "old_names": ["tmaskpac", ]},
    "vmaskatl": {"dims": ["y_r", "x_c"], "old_names": ["tmaskatl", ]},
    "vmaskind": {"dims": ["y_r", "x_c"], "old_names": ["tmaskind", ]},
    "vmaskpac": {"dims": ["y_r", "x_c"], "old_names": ["tmaskpac", ]},
    "fmaskatl": {"dims": ["y_r", "x_r"], "old_names": ["tmaskatl", ]},
    "fmaskind": {"dims": ["y_r", "x_r"], "old_names": ["tmaskind", ]},
    "fmaskpac": {"dims": ["y_r", "x_r"], "old_names": ["tmaskpac", ]}
}

rename_dims = {
    "time_counter": "t",
    "Z": "z",
    "Y": "y",
    "X": "x"
}

z_dims = (
    "z_c",
    "z_l",
    "z",
    "deptht",
    "depthu",
    "depthv",
    "depthw"
)

t_dims = (
    "t",
    "time_counter"
)
