"""Regression test of the input file and of the core sample size (needs the data file).

Fails, rather than skipping, when the data file is absent: the pipeline cannot be reproduced
without it (place the AmeriFlux FLUXNET CSV in data/ or set NPW_DATA_DIR, see docs/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C


def test_data_file_checksum():
    assert C.HH_FILE.exists(), f"data file not found: {C.HH_FILE} (set NPW_DATA_DIR or copy the file to data/)"
    sha = C.file_sha256(C.HH_FILE)
    assert sha == C.HH_FILE_SHA256, f"sha256 {sha} != reference {C.HH_FILE_SHA256}"


def test_core_sample_count():
    import load as L
    df = L.add_derived(L.read_hh())
    n = int(L.mask_core(df).sum())
    assert n == C.CORE_N_REGRESSION, n
    flags = L.data_flags(df, L.mask_core(df))
    assert flags["G_F_MDS_present"] == 0
    assert flags["SW_OUT_gt_SW_IN_when_SW_IN_gt_50"] == 0
    assert flags["SW_OUT_negative_day"] == 0
    # dawn-relaxed sample is a superset of the core sample
    assert int((L.mask_core_dawn_relaxed(df) & ~L.mask_core(df)).sum()) > 0
    assert bool((L.mask_core(df) & ~L.mask_core_dawn_relaxed(df)).sum() == 0)


if __name__ == "__main__":
    test_data_file_checksum(); print("PASS test_data_file_checksum")
    test_core_sample_count(); print("PASS test_core_sample_count")
