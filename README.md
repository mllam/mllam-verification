# mllam-verification
Verification of neural-lam, e.g. performance of the model relative to the truth, persistence, etc.

## General API patterns

- Every plot function should have ax = None as input argument. If no axes is provided, a new figure with appropriate settings/features (e.g. coastlines for maps) should be created. Otherwise, the plot should be added to the provided axes on top of the existing plot with a zorder = 10*n, where n is an integer. One can also specify zorder as input argument to the plot function to place the plot at a specific place in the plot hierarchy.
- Every plot function should have an include_persistence input argument, that defaults to True if it is possible to add persistence for the given plot type, and False if not. If include_persistence = True , but the plot doesn't support plotting the persistence an error should be raised.
- Every plot function should take the metric to compute and plot as input argument.
- The functions shall be callable from JupyterNotebook/Python script and the CLI.
- The top-level plot functions should be named according to the plot they produce. They should not directly contain the logic to actually compute the metrics, but instead call other comput functions to do that.
- The package shall support specifying plot settings, output path for saving plots, etc. in a config file

### Example on python functions

In [mllam_verification.plot](https://github.com/mllam/mllam-verification/pull/2/files#diff-15c7e6b996e82bcc0f02abdd0aa702dcc8d0afae552cec7872cbc1005081c896) there is an example on a plot function that plots the single-metric-timeseries plot

This plot function will call a calculate_{metric} function (located in the metrics.py module), that could look like (for metric=rmse) [this](https://github.com/mllam/mllam-verification/pull/2/files#diff-83955882ad971de54d93c75790537d89f01aaea7969c6e31109456d248aa1a20) with a persistence calculation function as shown later in that file.

These two functions will make use of a RMSE function located in the [statistics.py](https://github.com/mllam/mllam-verification/pull/2/files#diff-938dea8677e64061150529b5d1ca7753c260b27568d2d7be0a911edefe3fbd6a) module. The statistics.py functions will call functions from the `scores` python package where possible and add relevant cf compliant cell_methods.

## Python API
The mllam_verification package should be structured according to this directory structure. As an example, the above plot function plot_single_metric_timeseries will be located in mllam_verification/plot.py .

```
.
├── mllam_verification
│   ├── operations
│   │   ├── __init__.py
│   │   ├── dataset_manipulation.py     # Contains functions for dataset manipulation e.g. aligning shapes etc.
│   │   ├── loading.py                  # Contains functions for loading data
│   │   ├── saving.py                   # Contains functions for saving data and plots
│   │   ├── plot_utils.py               # Contains utility functions for plotting e.g. color maps, figure instanciation and formating etc.
│   │   ├── statistics.py               # Contains functions for computing statistics e.g. mean, std, etc.
│   │   └── metrics.py                  # Contains functions for computing metrics e.g. mean squared error, etc.
│   ├── __init__.py
│   ├── __main__.py                     # Entry point of the package
│   ├── argument_parser.py              # Contains CLI argument parser
│   ├── config.py                       # Contains config file parser
│   └── plot.py                         # Main script for producing plots
└── tests
    ├── conftest.py
    ├── unit
    │   ├── conftest.py
    │   └── ...
    └── integration
        ├── conftest.py
        └── ...
├── pdm.lock
├── pyproject.toml
├── example.yaml                        # Example config file
└── README.md
```

## CLI API
The package shall be callable with the following arguments
```bash
mllam_verification -r/--reference /path/to/ds_reference -p/--prediction /path/to/ds_prediction -m/--metric <metric_name> --var <variable-name(s)> --plot <name-of-plot(s)> --save-plots --save-dir /path/to/output
```

## Supported plots
The following is a first draft on the plots we want to make available in the mllam-verification package and what they support:
| Name                        | Plot function name                  | Example          | Grouped | Elapsed | UTC | Multi | Multi model | Multi variable | Point | Regular |
|-----------------------------|-------------------------------------|------------------|---------|---------|-----|-------|-------------|---------------|-------|---------|
| Single metric timeseries    | `plot_single_metric_timeseries`    | ![single_metric_timeseries_example](./docs/_images/single_metric_timeseries_example.png) | ✅¹| ✅¹| ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Single metric hovmöller     | `plot_single_metric_hovmoller`     | ![single_metric_hovmoller_example](./docs/_images/single_metric_hovmoller_example.png) | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ |
| Single metric gridded map   | `plot_single_metric_gridded_map`   | ![single_metric_gridded_map](./docs/_images/single_metric_gridded_map_example.png) | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Single metric point map     | `plot_single_metric_point_map`     | ![single_metric_point_map](./docs/_images/single_metric_point_map_example.png) | ✅ | ❌ | ❌ | ✅² | ✅² | ❌ | ✅ | ✅ |

¹ without persistence\
² maybe not a good idea e.g. if points overlap in grid.
