# XORCA

[![pipeline status](https://git.geomar.de/willi-rath/xorca/badges/master/pipeline.svg)](https://git.geomar.de/willi-rath/xorca/commits/master)
[![coverage report](https://git.geomar.de/willi-rath/xorca/badges/master/coverage.svg)](https://git.geomar.de/willi-rath/xorca/commits/master)


## What is this about?

XORCA brings [XGCM](https://xgcm.readthedocs.io) and
[Xarray](https://xarray.pydata.org) to the ORCA grid.  (It actually brings all
this to NEMO output.  But [xnemo was already
taken](https://github.com/serazing/xnemo).)

It allows for opening all output files from a model run into one Xarray dataset
that is understood by XGCM.  With this, grid-aware differentiation and
integration / summation is possible.


### Example: Barotropic stream function in XXX lines

```python
import xarray as xr
import xgcm
from xorca.lib import preprocess_orca

original_ds = xr.open_mfdataset(list_of_all_model_output_files)
ds = preprocess_orca(path_to_mesh_mask_file, original_ds)

grid = xgcm.Grid(ds, periodic=["Y", "X"])

U_bt = (ds.vozocrtx * ds.e3u).sum("z_c")

psi = grid.cumsum(- U_bt * ds.e2u, "Y") / 1.0e6

psi.mean("t").plot(size=9);
```

![barotropic stream function](doc/images/barotropic_stream_function.png)


### More examples

See the example notebook for hints on where this might end:
[notebooks/calculate_psi_speed_and_amoc.ipynb](notebooks/calculate_psi_speed_and_amoc.ipynb).
