# Surface entropy budget of BR-Npw (northern Pantanal): code and derived tables

This repository holds the code and the derived tables for the article "Surface entropy production of a seasonally flooded woody savanna in the northern Pantanal: absorbed-shortwave scaling and the energy-balance closure residual as the dominant uncertainty" (submitted to *Theoretical and Applied Climatology*).

It computes the half-hourly surface entropy budget of the AmeriFlux site BR-Npw from the FLUXNET (ONEFlux v1.3 r1) product for 2015–2016, along with the closure scenarios, the moving-block bootstrap and the sensitivity table reported in the article and its Supplementary Information.

The input data are not included. They are available from AmeriFlux under CC-BY-4.0 (see "Input data" below).

## Contents

```
code/
  config.py            paths, constants, selection rules, options; config and code hashes
  load.py              reads the FLUXNET half-hourly file and applies the core selection
  entropy.py           radiation and heat entropy fluxes (eqs. 1-14 of the article)
  closure.py           closure scenarios S0-S3
  storage.py           illustrative soil and water storage scenarios
  stats.py             stratified estimator and moving-block bootstrap
  figures.py           figures (Figs. 1-4, S2, S3)
  run_all.py           main pipeline: writes results/tables, results/figures, results/run_log.txt
  addons/rerun_2026_11_01.py   additions used in the final version (F/T convention row,
                               eps_atm = 0.85 row, S3 with per-period closure ratios, estimators,
                               RMA closure slopes, nighttime SR); appends to the tables and
                               writes results/run_log_rerun_2026-11-01.txt
  addons/figures_v10.py        Fig. S1 (sensitivity chart) with the row names of Table S4
  tests/test_entropy.py        18 unit tests of the entropy functions
  tests/test_select.py         2 tests: data checksum and core sample count (25,662)
results/tables/        the 23 derived tables (CSV) used for every number in the article
data/                  empty; put the AmeriFlux file here
```

Comments in the code mention `docs/`. Those notes are internal and not part of this repository. The article and its Supplementary Information describe the methods in full.

## Software

The results were produced with:

- Python 3.13.17
- pandas 3.0.5
- numpy 2.5.3
- matplotlib 3.11.2
- pytest (only for running the tests)

No other packages are needed. Other recent versions will probably work, but the hashes below are guaranteed only for these versions.

## Input data

1. Download the AmeriFlux FLUXNET product for BR-Npw, version v1.3_r1 (2013–2017), from the AmeriFlux data portal (https://ameriflux.lbl.gov), DOI https://doi.org/10.17190/AMF/1881563. Downloading requires a free AmeriFlux account and acceptance of the CC-BY-4.0 data policy.
2. Unzip it and place the half-hourly file `AMF_BR-Npw_FLUXNET_FLUXMET_HH_2013-2017_v1.3_r1.csv` in `data/`, or set the environment variable `NPW_DATA_DIR` to the folder that holds it.
3. The pipeline checks the file's sha256 and records the result in the run log. Expected value:
   `48fbd354fc9c402e04881eedf018ddda011d9fb4981f4714caacd2e231c896f1` (114,449,730 bytes, 87,648 records).

If AmeriFlux publishes a new release of the product, the checksum will differ and the numbers may change.

## Running

From the `code/` folder:

```
python3 -I run_all.py
python3 -I addons/rerun_2026_11_01.py
python3 -I -m pytest -q tests/
```

Run the steps in this order. `run_all.py` rewrites `t4_sensitivity.csv` and Fig. S1, so the add-on must run after it every time. The add-on then appends its rows and redraws Fig. S1 (through `addons/figures_v10.py`).

A full run takes under a minute on a laptop. The expected test result is `20 passed`.

Output goes to `results/` (tables, figures, run logs) next to `code/`. Intermediate files go to `scratch/`, or to the folder set in `NPW_SCRATCH_DIR`.

## Expected provenance and hashes

The run logs should report:

- config hash `d96faf50403b`
- code hash `a5a3ec682769`, the sha256 of `code/*.py`; the add-ons and tests are not included in it
- add-on sha256 starting with `b56071103a28`
- seed `20261030`, 1000 bootstrap replicates, 7-day blocks
- core sample n = 25,662

The sha256 of each derived table:

| Table | sha256 |
|---|---|
| t0_data_coverage.csv | 028669071937e8b2761e14f21377f05ba65ca4a2cf9831cce7f3eea3dc9542ce |
| t0b_selection_fraction_by_slot.csv | 381e1fdac2c8fa38d1951ffbd6b659ed3b97ab98160426b5b92d0334a3e33c4c |
| t0c_selection_bias_core_vs_excluded.csv | 01f969d195e7b8cf7de203759adbd9f9bdd295136722d72b5c5d7b8ca047c6e3 |
| t0d_swin_qc_counts.csv | cad3addc4bd96904ffc1b5100ccd3ad1c27d1e999dac8093ae6198e0835b2e0c |
| t1_energy_fluxes.csv | d48b355416532d000ccad3b0c3684697ff942b9bbd7ab82ed63a4f9a7515f2a5 |
| t2_entropy_budget.csv | 72496ae4a84e1ff0639b2d6a7f2ceec1be5ca69f7c58eaaad7c976aa9f03121e |
| t2b_bootstrap_differences.csv | 37618becf027ba3f0961a12d2fe99953975cdcc47aa82fca65d6524803827b5f |
| t2c_analytic_decomposition.csv | f41a0ef7881187714cff38d80c434bd815a4e02e7f7e8aa02d245ebe924b7a01 |
| t2d_matched_month_differences.csv | 1a9e46ac1437ba6bdbf563e901db378b880cbf6fedfafd00214ab7a6726bbc66 |
| t2e_block_length_sensitivity.csv | 65a9a84c903904cbf75f78eb5c2e655c50b78f91f7238ea176a8ceab6989f1f0 |
| t2f_bootstrap_estimators.csv | ec92fa9474da2e9ddabd683e27b1a834255b0a982e5d76123be33a5af5e66aef |
| t3_closure_scenarios.csv | 3dff5f8473d50e4879e34540e1adfe062c32c7b09b5175348bcb5cbaf1c73e0d |
| t3b_closure_S3_per_period_cr.csv | 9ef6c95cd5defca82da6ad336767b39fbfda4395f97611b008a9080de7c25ad6 |
| t3c_closure_rma_and_abs_residual.csv | d5e2e5decd15c58066993de563077d0cab3483cff1337d35b1b2cd29fd7457e6 |
| t4_sensitivity.csv | e3d712b23071e33bab00fdcaf9613c24ba5fed8817cedc275f8e02d473dafbe5 |
| t4b_sr_night_midnight.csv | 762515e1132bdbd5ec518d40040f0f78e4e26c3e0ca778d26bb93557f41b8cb7 |
| t4c_ft_convention_check.csv | 815843fadbfb23e3c477990e1ae289de9fc5b9d8fe194ea81c6082ac70c5dba5 |
| t5_monthly_climatology.csv | d6a31d76603de76b31b3b159c05914db38ace837b161dffe9062ec424c60a18e |
| t5_monthly_climatology_by_year.csv | 72298773e7aeef9469e75554d1d88a469782c5378db989d057851fa35a310fa6 |
| t6_literature_comparison.csv | ac85127967b6e3813432638793652261adce9e5ce705cb7a3027f828bd74c038 |
| t7_diurnal_composites.csv | e4052e2401a3ced2fa1a319d9a3c22ee335aa8eb5c123cbeed51fa729a8d6ddf |
| t8_midday_albedo_by_month.csv | af39e5cab29b16255da5698754b04489f2c7806ba5967a3b9d0658a7695f0180 |
| t9_daily_means_complete_days.csv | dc799f1f70b8d47485588c3c1a0d740969f89987a43cc23349d56ef4dbce2654 |

To check them on Linux, run `sha256sum results/tables/*.csv`.

On 1 November 2026 a clean copy of this code was run from scratch with the steps above. All 23 tables matched these hashes byte for byte, and the PNG figures matched the published ones pixel for pixel.

Notes:

- The figures were drawn with the Inter font, which was the default of the matplotlib installation used. With another default font the figures look slightly different, but the numbers are the same.
- The submitted figure files carry panel letters (a, b, c, d), which were added after rendering. The PNGs that the code writes have no panel letters.
- `t6_literature_comparison.csv` holds values taken from the published studies cited in the article, not values computed here.

## Licence

- Code (`code/`): MIT licence.
- Derived tables (`results/tables/`): CC-BY-4.0.
- Input data: not included. AmeriFlux FLUXNET product under CC-BY-4.0, which requires citation of the site dataset (DOI above) and the AmeriFlux and FLUXNET acknowledgement texts.

See `LICENSE` and `CITATION.cff`.
