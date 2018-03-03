import xgcm


def calculate_psi(ds):
    grid = xgcm.Grid(ds, periodic=["Y", "X"])
    U_bt = (ds.vozocrtx * ds.e3u).sum("z_c")
    psi = grid.cumsum(- U_bt * ds.e2u, "Y") / 1.0e6
    psi -= psi.isel(y_r=-1, x_r=-1)  # normalize upper right corner
    psi = psi.rename("psi")
    return psi
