import xgcm


def calculate_moc(ds, region=""):
    grid = xgcm.Grid(ds, periodic=["Y", "X"])

    vmaskname = "vmask" + region
    mocname = "moc" + region
    latname = "lat_moc" + region

    weights = ds[vmaskname] * ds.e3v * ds.e1v

    Ve3 = weights * ds.vomecrty

    # calculate indefinite vertical integral of V from bottom to top, then
    # integrate zonally, convert to [Sv], and rename to region
    moc = grid.cumsum(Ve3, "Z", to="left", boundary="fill") - Ve3.sum("z_c")
    moc = moc.sum("x_c")
    moc /= 1.0e6
    moc = moc.rename(mocname)

    # calculate the weighted zonal and vertical mean of latitude
    lat_moc = ((weights * ds.llat_rc).sum(dim=["z_c", "x_c"]) /
               (weights).sum(dim=["z_c", "x_c"]))
    moc.coords[latname] = (["y_r", ], lat_moc.data)

    # also copy the relevant depth-coordinates
    moc.coords["depth_l"] = ds.coords["depth_l"]

    return moc


def calculate_psi(ds):
    grid = xgcm.Grid(ds, periodic=["Y", "X"])

    U_bt = (ds.vozocrtx * ds.e3u).sum("z_c")

    psi = grid.cumsum(- U_bt * ds.e2u, "Y") / 1.0e6
    psi -= psi.isel(y_r=-1, x_r=-1)  # normalize upper right corner
    psi = psi.rename("psi")

    return psi


def calculate_speed(ds):
    grid = xgcm.Grid(ds, periodic=["Y", "X"])

    U_cc = grid.interp(ds.vozocrtx, "X", to="center")
    V_cc = grid.interp(ds.vomecrty, "Y", to="center")

    speed = (U_cc**2 + V_cc**2)**0.5

    return speed
