"""Test reading NEMO v4 output data"""

from pathlib import Path
import pytest
from zenodo_get import zenodo_get

from xorca.lib import load_xorca_dataset


data_path = '_example_data/NEMO_v4.0.0'
# Download the test data from zenodo
zenodo_get(['4618128', f'--output-dir={data_path}'])
data_path = Path(data_path)

update_orca_variables={
    "thetao": {"dims": ["t", "z_c", "y_c", "x_c"]},
    "so": {"dims": ["t", "z_c", "y_c", "x_c"]},
    "uo": {"dims": ["t", "z_c", "y_c", "x_r"]},
    "vo": {"dims": ["t", "z_c", "y_r", "x_c"]},
    "woce": {"dims": ["t", "z_l", "y_c", "x_c"]},
    "tos_month": {"dims": ["t", "y_c", "x_c"]}
}


def test_nemo_v_4_0_0_one_point_at_a_time():
    """
    In NEMO 4.0.0 the vertical coordinate is not called
    'z' but 'nav_lev'. Testing that the 'nav_lev' name is understood

    Opening one point at a time
    """
    for i in ['T', 'U', 'V', 'W']:
        data_files = list(data_path.glob(f"BASIN_1d_00010101_00010103_grid_{i}*"))
        aux_files = list(data_path.glob("mesh*.nc"))
        
        data_set = load_xorca_dataset(
            data_files=data_files, aux_files=aux_files,
            update_orca_variables=update_orca_variables
        )


def test_nemo_v_4_0_0():
    """
    In NEMO 4.0.0 the vertical coordinate is not called
    'z' but 'nav_lev'. Testing that the 'nav_lev' name is understood
    """
    data_files = list(data_path.glob("BASIN*.nc"))
    aux_files = list(data_path.glob("mesh*.nc"))

    data_set = load_xorca_dataset(
        data_files=data_files, aux_files=aux_files,
        update_orca_variables=update_orca_variables
    )
    assert ('thetao' in data_set)
    assert ('vo' in data_set)
