# mllam-verification

[![Linting](https://github.com/mllam/mllam-verification/actions/workflows/ci-pre-commit.yml/badge.svg?branch=main)](https://github.com/mllam/mllam-verification/actions/workflows/ci-pre-commit.yml)

This package exposes various plotting routines for verification of prediction data relative to some reference data. The plotting functions are meant to be used in JupyterNotebooks or Python scripts, and thus no CLI is provided.

The plotting routines accept a handle to a callable, which is used for calculating the statistical metric to plot. Let's take the `plot_single_metric_timeseries` function as an example:

```python
from mllam_verification.plot import plot_single_metric_timeseries
from mllam_verification.operations.statistics import rmse

ax = plot_single_metric_timeseries(
    da_reference,
    da_prediction,
    time_axis="UTC",
    stats_operation=rmse,
    hue="analysis_time",
)
```

The `plot_single_metric_timeseries` function will call the `rmse` function to compute the RMSE between the reference and prediction data before plotting. An axes object is returned, which can be used to add more plots to the same figure, or to adjust the axes layout.

The following table outlines the different types of plots one can produce with the mllam-verification package and what they support. More to come!
| Name                        | Plot function name                  | Example          | Grouped | Elapsed | UTC | Multi | Multi model | Multi variable | Point | Regular |
|-----------------------------|-------------------------------------|------------------|---------|---------|-----|-------|-------------|---------------|-------|---------|
| Single metric timeseries    | `plot_single_metric_timeseries`    | ![single_metric_timeseries_example](./docs/_images/single_metric_timeseries_example.png) | ✅¹| ✅¹| ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Single metric hovmöller     | `plot_single_metric_hovmoller`     | ![single_metric_hovmoller_example](./docs/_images/single_metric_hovmoller_example.png) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Single metric gridded map   | `plot_single_metric_gridded_map`   | ![single_metric_gridded_map](./docs/_images/single_metric_gridded_map_example.png) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
~~single metric point map~~²     | ~~`plot_single_metric_point_map`~~ ²     | ![single_metric_point_map](./docs/_images/single_metric_point_map_example.png) | ✅ | ❌ | ❌ | ✅² | ✅² | ❌ | ✅ | ✅ |

¹ without persistence\
² not supported yet, but soon to come\
³ maybe not a good idea e.g. if points overlap in grid.

## Developing `mllam-data-prep`

To work on developing `mllam-verification` it is easiest to install and manage the dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/). To get started clone your fork of [the main repo](https://github.com/mllam/mllam-verification) locally:

```bash
git clone https://github.com/<your-github-username>/mllam-verification
cd mllam-verification
```

You can now e.g. run tests of `mllam-verification` with uv directly
```bash
uv run pytest
```
or you can first create a virtualenv and install the dependencies
```bash
uv venv
uv sync --all-extras
uv run pytest
```

All the linting is handelled by `pre-commit` which can be setup to automatically be run on each `git commit` by installing the git commit hook:

```bash
uv run pre-commit install
```

Then branch, commit, push and make a pull-request :)

## Repo structure
```
.
├── mllam_verification
│   ├── operations
│   │   ├── __init__.py
│   │   ├── array_handling.py           # Contains helper functions for array operations
│   │   └── statistics.py               # Contains functions for computing statistics e.g. mean, std, etc.
│   ├── __init__.py
│   ├── __main__.py                     # Entry point of the package
│   ├── plot.py                         # Main script for producing plots
│   └── validation.py                   # Contains functionality to validate function arguments
├── uv.lock
├── pyproject.toml
└── README.md
```
