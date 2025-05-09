from datetime import datetime
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import xarray as xr

import mllam_verification.operations.statistics as mlverif_stats


def plot_single_metric_timeseries(
    da_reference: xr.DataArray,
    da_prediction: xr.DataArray,
    stats_operation: Callable = mlverif_stats.rmse,
    axes: Optional[plt.Axes] = None,
    time_axis: Literal["groupedby.{grouping}.{group}", "elapsed", "UTC"] = "elapsed",
    time_operation: Optional[Callable] = None,
    time_op_kwargs: Optional[dict] = {},
    include_persistence: Optional[bool] = True,
    hue: Optional[str] = "datasource",
    xarray_plot_kwargs: Optional[dict] = {},
) -> plt.Axes:
    """Plot a single-metric-timeseries diagram for a given metric.

    The metric is calculated from da_reference and da_prediction, which should
    have one of the following specifications:

    A) For data with non-regular grid:

    Dimensions: [start_time, elapsed_forecast_duration, grid_index, datasource]
    Coordinates:
    - start_time:
        the start time as a datetime object
    - elapsed_forecast_duration:
        the elapsed forecast duration as a timedelta object
    - grid_index:
        the index of the gridpoint
    - datasource:
        the source of the data, e.g. model name, persistence, etc.

    B) For data with regular grid:

    Dimensions: [start_time, elapsed_forecast_duration, x, y, datasource]
    Coordinates:
    - start_time:
        the analysis time as a datetime object
    - elapsed_forecast_duration:
        the elapsed forecast duration as a timedelta object
    - x:
        the x coordinate of the gridpoint
    - y:
        the y coordinate of the gridpoint
    - datasource:
        the source of the data, e.g. model name, persistence, etc.

    The `start_time` dimension can be omitted from the dataarray. If it is present,
    the metric will be averaged along the `start_time` dimension before plotting.

    In case B), the `x` and `y` dimensions will be stacked into a single `grid_index`
    dimension before plotting.

    Parameters
    ----------
    da_reference : xr.DataArray
        Reference dataarray.
    da_prediction : xr.DataArray
        Prediction dataarray.
    variable : str
        Variable to calculate metric of.
    stats_operation : Callable, optional
        Statistics operation to calculate the metric, by default mlverif_stats.rmse
    axes : plt.Axes, optional
        Axes to plot on, by default None
    include_persistence : bool, optional
        Whether to include persistence in the plot, by default True
    hue : str, optional
        Hue to plot on, by default "datasource"
    xarray_plot_kwargs : dict, optional
        Additional arguments to pass to xarray's plot function, by default {}

    Returns
    -------
    plt.Axes
        Axes with the plot added.
    """
    if include_persistence:
        da_reference, da_prediction = mlverif_stats.add_persistence_to_dataarray(
            da_reference, da_prediction
        )

    time_axis_items = time_axis.split(".")
    groupby: str | None = None
    group: str | int | None = None
    if len(time_axis_items) >= 2:
        if time_axis_items[0] != "groupedby":
            raise ValueError(
                f"Expected 'time_point' to start with 'groupedby', got {time_axis}"
            )
        groupby = time_axis_items[1]
        if len(time_axis_items) > 2:
            try:
                group = int(time_axis_items[-1])
            except ValueError:
                group = time_axis_items[-1]

    # Stack the x and y dimensions into a single grid index if necessary
    if "x" in da_prediction.dims and "y" in da_prediction.dims:
        da_prediction = da_prediction.stack(grid_index=["x", "y"])
    if "x" in da_reference.dims and "y" in da_reference.dims:
        da_reference = da_reference.stack(grid_index=["x", "y"])
    # Apply statistical operation
    da_metric: xr.DataArray | xr.core.groupby.DataArrayGroupBy = stats_operation(
        da_reference,
        da_prediction,
        reduce_dims=["grid_index"],
        groupby=f"time.{groupby}" if groupby is not None else groupby,
    )
    if time_operation is not None:
        da_metric = time_operation(da_metric, **time_op_kwargs)

    if group is not None:
        if isinstance(da_metric, xr.DataArray):
            da_metric: xr.DataArray = da_metric.sel({groupby: group})
        elif isinstance(da_metric, xr.core.groupby.DataArrayGroupBy):
            da_metric: xr.DataArray = da_metric[group]
        else:
            raise ValueError(
                "da_metric must be an xr.DataArray or xr.core.groupby.DataArrayGroupBy"
            )

    if axes is None:
        _, axes = plt.subplots()

    if hue not in da_metric.coords:
        raise ValueError(
            f"DataArray does not contain a coordinate named {hue}, "
            + "please use a different coordinate as hue"
        )
    da_metric.plot.line(
        ax=axes, hue=hue, **xarray_plot_kwargs if xarray_plot_kwargs is not None else {}
    )

    return axes


def plot_single_metric_gridded_map(
    da_reference: xr.DataArray,
    da_prediction: xr.DataArray,
    stats_operation: Callable = mlverif_stats.difference,
    axes: Optional[plt.Axes] = None,
    time_selection: Optional[Literal["groupedby.{grouping}.{group}"] | datetime] = None,
    time_operation: Optional[Callable] = None,
    xarray_plot_kwargs: Optional[dict] = None,
):
    """Plot a single-metric-gridded-map diagram for a given metric.

    The metric is calculated from da_reference and da_prediction, which should
    have the following specification:

    Dimensions: [start_time, x, y]
    Data variables:
    - state [start_time, x, y]:
    Coordinates:
    - start_time:
        the analysis time as a datetime object
    - x:
        the x coordinate of the gridpoint
    - y:
        the y coordinate of the gridpoint

    The `start_time` dimension can be omitted from the dataarray. If it is present,
    the metric will be averaged along the `start_time` dimension before plotting.

    Parameters
    ----------
    da_reference : xr.DataArray
        Reference dataarray.
    da_prediction : xr.DataArray
        Prediction dataarray.
    variable : str
        Variable to plot.
    axes : plt.Axes, optional
        Axes to plot on, by default None
    xarray_plot_kwargs : dict, optional
        Additional arguments to pass to xarray's plot function, by default {}

    Returns
    -------
    plt.Axes
        Axes with the plot added.
    """

    # Check if dataarrays have necessary dimensions
    if "x" not in da_prediction.dims or "y" not in da_prediction.dims:
        raise ValueError(
            "Prediction dataarray must have x and y dimensions to plot a gridded map"
        )
    if "x" not in da_reference.dims and "y" not in da_reference.dims:
        raise ValueError(
            "Reference dataarray must have x and y dimensions to plot a gridded map"
        )

    groupby: str | None = None
    group: str | int | None = None
    if isinstance(time_selection, datetime):
        da_reference = da_reference.sel(time=time_selection)
        da_reference = da_reference.sel(time=time_selection)
    elif isinstance(time_selection, str):
        time_selection_items = time_selection.split(".")
        if len(time_selection_items) < 3:
            raise ValueError(
                "Expected 'time_point' to have format "
                f"'groupedby.{{grouping}}.{{group}}', got {time_selection}"
            )
        if time_selection_items[0] != "groupedby":
            raise ValueError(
                f"Expected 'time_point' to start with 'groupedby', got {time_selection}"
            )
        groupby = time_selection_items[1]
        try:
            group = int(time_selection_items[-1])
        except ValueError:
            group = time_selection_items[-1]

    elif time_selection is not None:
        raise TypeError(
            f"Expected 'time_selection' to be either None, a datetime or a string with format 'groupedby.{{grouping}}.{{group}}', got {type(time_selection)}"
        )

    # Apply operations
    da_metric: xr.DataArray | xr.core.groupby.DataArrayGroupBy = stats_operation(
        da_reference,
        da_prediction,
        groupby=f"time.{groupby}" if groupby is not None else groupby,
    )
    if time_operation is not None:
        da_metric = time_operation(da_metric)

    # Select relevant data:
    if group is not None:
        if isinstance(da_metric, xr.DataArray):
            da_metric: xr.DataArray = da_metric.sel({groupby: group})
        elif isinstance(da_metric, xr.core.groupby.DataArrayGroupBy):
            da_metric: xr.DataArray = da_metric[group]
        else:
            raise ValueError(
                "da_metric must be an xr.DataArray or xr.core.groupby.DataArrayGroupBy"
            )

    if axes is None:
        _, axes = plt.subplots()

    if "time" in da_metric.dims and da_metric.sizes["time"] == 1:
        da_metric = da_metric.isel(time=0)
    if "time" in da_metric.dims and da_metric.sizes["time"] >= 1:
        if groupby is not None:
            raise ValueError(
                "'time_operation' must be provided to reduce dimensionality of "
                "grouped dataarray da_metric before plotting."
            )
        raise ValueError(
            "Please select a specific time to plot with 'time_selection=<datetime>'"
        )
    if len(da_metric.dims) != 2:

        raise ValueError(
            "Metric DataArray must have 2 dimensions (x, y) to plot a gridded map"
        )
    da_metric.plot.pcolormesh(
        x="x",
        y="y",
        ax=axes,
        **xarray_plot_kwargs if xarray_plot_kwargs is not None else {},
    )

    return axes


def plot_single_metric_hovmoller(
    da_reference: xr.DataArray,
    da_prediction: xr.DataArray,
    preserve_dim: str,
    stats_operation: Callable = mlverif_stats.rmse,
    axes: Optional[plt.Axes] = None,
    xarray_plot_kwargs: Optional[dict] = None,
):
    """Plot a single-metric-hovmoller diagram for a given metric.

    The plot will have time on the x-axis, and the `preserve_dim`
    dimension on the y-axis.

    The metric is calculated from da_reference and da_prediction, which should
    have one of the following specifications:

    Dimensions: [start_time, elapsed_forecast_duration, `spatial_dim`, ...]
    Coordinates:
    - start_time:
        the analysis time as a datetime object
    - elapsed_forecast_duration:
        the elapsed forecast duration as a timedelta object
    - `spatial_dim`:
        the coordinate of the spatial dimension to be plotted up the y-axis
    - ...:
        Any other dimensions, which will be reduced along

    The `start_time` dimension can be omitted from the dataarray. If it is present,
    the metric will be averaged along the `start_time` dimension before plotting.

    Parameters
    ----------
    da_reference : xr.DataArray
        Reference dataarray.
    da_prediction : xr.DataArray
        Prediction dataarray.
    variable : str
        Variable to plot.
    preserve_dim : str
        Dimension to preserve along the y-axis.
    stats_operation : Callable, optional
        Statistics operation to calculate the metric, by default mlverif_stats.rmse
    axes : plt.Axes, optional
        Axes to plot on, by default None
    xarray_plot_kwargs : dict, optional
        Additional arguments to pass to xarray's plot function, by default {}

    Returns
    -------
    plt.Axes
        Axes with the plot added.
    """

    # Check if dataarrays have necessary dimensions
    if preserve_dim not in da_prediction.dims or preserve_dim not in da_reference.dims:
        raise ValueError(
            "Prediction and reference dataarrays must have `preserve_dim`"
            " dimension to plot hövmöller diagram."
        )

    # Apply statistical operation
    ds_metric: xr.DataArray = stats_operation(
        da_reference,
        da_prediction,
        preserve_dims=[preserve_dim, "start_time", "elapsed_forecast_duration"],
    )
    if "start_time" in ds_metric.dims:
        ds_metric: xr.DataArray = mlverif_stats.mean(ds_metric, dim=["start_time"])

    if axes is None:
        _, axes = plt.subplots()

    # Check if dimensions are present
    if len(ds_metric.dims) != 2:
        raise ValueError(
            "Metric dataarray must have 2 dimensions (`preserve_dim`, "
            "elapsed_forecast_duration) to plot a gridded map"
        )

    ds_metric.plot.pcolormesh(
        x="elapsed_forecast_duration",
        y=preserve_dim,
        ax=axes,
        **xarray_plot_kwargs if xarray_plot_kwargs is not None else {},
    )

    return axes
