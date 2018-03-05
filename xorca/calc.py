"""Calculations with grid-aware data sets."""

import xgcm


def calculate_moc(ds, region=""):
    """Calculate the MOC.

    Parameters
    ----------
    ds : xarray dataset
        A grid-aware dataset as produced by `xorca.lib.preprocess_orca`.
    region : str
        A region string.  Examples: `"atl"`, `"pac"`, `"ind"`.
        Defaults to `""`.

    Returns
    -------
    moc : xarray data array
        A grid-aware data array with the moc for the specified region.  The
        data array will have a coordinate called `"lat_moc{region}"` which is
        the weighted horizontal and vertical avarage of the latitude of all
        latitudes for the given point on the y-axis.

    """
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
    """Calculate the barotropic stream function.

    Parameters
    ----------
    ds : xarray dataset
        A grid-aware dataset as produced by `xorca.lib.preprocess_orca`.

    Returns
    -------
    psi : xarray data array
        A grid-aware data array with the barotropic stream function in `[Sv]`.

    """
    grid = xgcm.Grid(ds, periodic=["Y", "X"])

    U_bt = (ds.vozocrtx * ds.e3u).sum("z_c")

    psi = grid.cumsum(- U_bt * ds.e2u, "Y") / 1.0e6
    psi -= psi.isel(y_r=-1, x_r=-1)  # normalize upper right corner
    psi = psi.rename("psi")

    return psi


def calculate_speed(ds):
    """Calculate speed on the central (T) grid.

    First, interpolate U and V to the central grid, then square, add, and take
    root.

    Parameters
    ----------
    ds : xarray dataset
        A grid-aware dataset as produced by `xorca.lib.preprocess_orca`.

    Returns
    -------
    speed : xarray data array
        A grid-aware data array with the speed in `[m/s]`.

    """
    grid = xgcm.Grid(ds, periodic=["Y", "X"])

    U_cc = grid.interp(ds.vozocrtx, "X", to="center")
    V_cc = grid.interp(ds.vomecrty, "Y", to="center")

    speed = (U_cc**2 + V_cc**2)**0.5

    return speed
