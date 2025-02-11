from types import ModuleType
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import mllam_verification.operations.statistics as mlverif_stats


def plot_single_metric_timeseries(
    ds_reference: xr.Dataset,
    ds_prediction: xr.Dataset,
    variable: str,
    stats_operation: ModuleType = mlverif_stats.rmse,
    axes: plt.Axes = None,
    include_persistence=True,
    hue: str = "datasource",
    xarray_plot_kwargs: dict = {},
):
    """Plot a single-metric-timeseries diagram for a given variable and metric.

    The metric is calculated from ds_reference and ds_prediction.

    Parameters
    ----------
    ds_reference : xr.Dataset
        Reference dataset.
    ds_prediction : xr.Dataset
        Prediction dataset.
    variable : str
        Variable to calculate metric of.
    metric : str
        Metric to calculate.
    axes : plt.Axes, optional
        Axes to plot on, by default None
    xarray_plot_kwargs : dict, optional
        Additional arguments to pass to xarray's plot function, by default {}
    """

    if include_persistence:
        ds_reference, ds_prediction = add_persistence_to_datasets(
            ds_reference, ds_prediction
        )

    # Stack the x and y dimensions into a single grid index if necessary
    if "x" in ds_prediction.dims and "y" in ds_prediction.dims:
        ds_prediction = ds_prediction.stack(grid_index=["x", "y"])
    if "x" in ds_reference.dims and "y" in ds_reference.dims:
        ds_reference = ds_reference.stack(grid_index=["x", "y"])

    ds_metric: xr.Dataset = stats_operation(
        ds_reference, ds_prediction, reduce_dims=["grid_index"]
    )
    ds_metric: xr.Dataset = mlverif_stats.mean(ds_metric, reduce_dims=["start_time"])

    if axes is None:
        _, axes = plt.subplots()

    if hue not in ds_metric.coords:
        raise ValueError(
            f"Dataset does not contain a coordinate named {hue}, "
            + "please use a different coordinate as hue"
        )

    ds_metric[variable].plot.line(ax=axes, hue=hue, **xarray_plot_kwargs)

    return axes


def add_persistence_to_datasets(
    ds_reference: xr.Dataset, ds_prediction: xr.Dataset
) -> Tuple[xr.Dataset, xr.Dataset]:
    """Add persistence datasource to datasets.

    Parameters
    ----------
    ds_reference : xr.Dataset
        Reference dataset.
    ds_prediction : xr.Dataset
        Prediction dataset.

    Returns
    -------
    Tuple[xr.Dataset, xr.Dataset]
        Reference and prediction datasets with persistence datasource added.
    """
    # Set persistence prediction as the reference shifted by 1
    ds_persistence_prediction = ds_reference.shift(elapsed_forecast_duration=1)
    # Save original datasources before concatenating
    reference_datasources = ds_reference["datasource"].values
    # Concatenate the datasets
    ds_reference = xr.concat([ds_reference, ds_reference], dim="datasource")
    ds_prediction = xr.concat(
        [ds_prediction, ds_persistence_prediction], dim="datasource"
    )
    # Update datasource coordinates
    ds_reference["datasource"] = ds_prediction["datasource"] = np.append(
        reference_datasources, "persistence"
    )

    return ds_reference, ds_prediction
