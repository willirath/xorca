# XORCA

master
[![pipeline status](https://git.geomar.de/willi-rath/xorca/badges/master/pipeline.svg)](https://git.geomar.de/willi-rath/xorca/commits/master)
[![coverage report](https://git.geomar.de/willi-rath/xorca/badges/master/coverage.svg)](https://git.geomar.de/willi-rath/xorca/commits/master)
|
develop
[![pipeline status](https://git.geomar.de/willi-rath/xorca/badges/develop/pipeline.svg)](https://git.geomar.de/willi-rath/xorca/commits/develop)
[![coverage report](https://git.geomar.de/willi-rath/xorca/badges/develop/coverage.svg)](https://git.geomar.de/willi-rath/xorca/commits/develop)


## What is this about?

XORCA brings [XGCM](https://xgcm.readthedocs.io) and
[Xarray](https://xarray.pydata.org) to the ORCA grid.  (It actually brings all
this to NEMO output.  But [xnemo was already
taken](https://github.com/serazing/xnemo).)

It allows for opening all output files from a model run into one Xarray dataset
that is understood by XGCM.  With this, grid-aware differentiation and
integration / summation is possible.


### Example: Calculate the barotropic stream function in 2 lines

After a short preamble which imports the package and loads the data:

```python
import xarray as xr
import xgcm
from xorca.lib import load_xorca_dataset

ds = load_xorca_dataset(data_files=list_of_all_model_output_files,
                        aux_files=list_of_mesh_mask_files
grid = xgcm.Grid(ds, periodic=["Y", "X"])
```

This is all that's needed to define and calculate the barotropic stream
function for all time steps:
```
U_bt = (ds.vozocrtx * ds.e3u).sum("z_c")
psi = grid.cumsum(- U_bt * ds.e2u, "Y") / 1.0e6
```

And this triggers the actual computation and produces the image:
```
psi.mean("t").plot(size=9);
```

![barotropic stream function](doc/images/barotropic_stream_function.png)

### More examples

See the example notebook for hints on where this might end:
[notebooks/calculate_psi_speed_and_amoc.ipynb](notebooks/calculate_psi_speed_and_amoc.ipynb).


## Installation

First, install all dependencies (assuming you have conda installed and in the
path):
```bash
curl \
    https://git.geomar.de/willi-rath/xorca/raw/master/environment.yml \
    -o xorca_environment.yml
conda env create -n xorca_env -f xorca_environment.yml
```

Then install XORCA:
```bash
source activate xorca_env
pip install git+https://git.geomar.de/willi-rath/xorca.git@master
```

To use, `source activate xorca_env` before, e.g., starting `jupyter notebook`.
