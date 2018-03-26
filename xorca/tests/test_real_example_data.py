"""Test reading the mesh masks."""

from pathlib import Path
import pytest

from xorca.lib import load_xorca_dataset


example_data_locations = {
    "ORCA025.L46.LIM2vp.JRA.XIOS2.KMS-T002": Path(
        "_example_data/ORCA025.L46.LIM2vp.JRA.XIOS2.KMS-T002/")}


@pytest.mark.skipif(
    not example_data_locations[
        "ORCA025.L46.LIM2vp.JRA.XIOS2.KMS-T002"].exists(),
    reason="Example files may not be present.")
def test_with_ORCA025_001():
    data_path = example_data_locations[
        "ORCA025.L46.LIM2vp.JRA.XIOS2.KMS-T002"]

    data_files = data_path.glob("ORCA025*_1m_*.nc")
    data_files = sorted(list(data_files), reverse=True)

    aux_files = data_path.glob("m*.nc")

    data_set = load_xorca_dataset(data_files=data_files,
                                  aux_files=aux_files)

    t = data_set.coords["t"]
    ts = t.sortby("t")

    assert all(td == tsd for td, tsd in zip(t.data, ts.data))
