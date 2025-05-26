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
    time_op_kwargs: Optional[dict] = None,
    include_persistence: Optional[bool] = False,
    hue: Optional[str] = "datasource",
    xarray_plot_kwargs: Optional[dict] = None,
) -> plt.Axes:
    """Plot a single-metric-timeseries diagram for a given metric.

    The metric is calculated from da_reference and da_prediction, which should
    have one of the following specifications:

    A) For data with non-regular grid and `time_axis="elapsed"`:

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

    B) For data with regular grid and `time_axis="elapsed"`:

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

    C) For data with non-regular grid and either `time_axis="UTC"`
        or `time_axis="groupedby.{grouping}.{group}"`:

    Dimensions: [time, grid_index, datasource]
    Coordinates:
    - time:
        the time coordinate as a datetime object
    - grid_index:
        the index of the gridpoint
    - datasource:
        the source of the data, e.g. model name, persistence, etc.

    D) For data with regular grid and either `time_axis="UTC"`
        or `time_axis="groupedby.{grouping}.{group}"`:

    Dimensions: [time, x, y, datasource]
    Coordinates:
    - time:
        the time coordinate as a datetime object
    - x:
        the x coordinate of the gridpoint
    - y:
        the y coordinate of the gridpoint
    - datasource:
        the source of the data, e.g. model name, persistence, etc.


    In case A) and B), the `start_time` dimension can be omitted from the
    dataarray. If it is present, one has to specify a `time_operation` callable,
    and possibly a `time_op_kwargs` dictionary with kwargs to pass to the callable,
    to reduce the dimensions of the dataarray before plotting.

    In case C) and D), if the provided `time_axis` is "groupedby.{grouping}", i.e.
    with no ".{group}" specified, in most cases a `time_operation` callable is
    required to reduce the xr.core.groupby.DataArrayGroupBy to an xr.DataArray.
    Only in the case where all data arrays of the xr.core.groupby.DataArrayGroupBy
    object have length one, this is not needed (the reduction is automatically
    handled by the function).

    In case B) and D), the `x` and `y` dimensions will be stacked into a single
    `grid_index` dimension before plotting.

    Parameters
    ----------
    da_reference : xr.DataArray
        Reference dataarray.
    da_prediction : xr.DataArray
        Prediction dataarray.
    stats_operation : Callable, optional
        Statistics operation to calculate the metric, by default mlverif_stats.rmse
    axes : plt.Axes, optional
        Axes to plot on, by default None
    time_axis : Literal["groupedby.{grouping}.{group}", "elapsed", "UTC"], optional
        Time axis to use when plotting, by default "elapsed"
    time_operation : Optional[Callable], optional
        Time operation to apply to the time dimension of the calculate da_metric
        dataarray before plotting, by default None
    time_op_kwargs : Optional[dict], optional
        kwargs to pass to the time operation, by default None
    include_persistence : bool, optional
        Whether to include persistence in the plot, by default True
        Only possible if `time_axis="utc"`
    hue : str, optional
        Hue to plot on, by default "datasource"
    xarray_plot_kwargs : dict, optional
        Additional arguments to pass to xarray's plot function, by default None

    Returns
    -------
    plt.Axes
        Axes with the plot added.
    """
    if include_persistence:
        if time_axis != "utc":
            raise ValueError("include_persistence is only possible if time_axis='utc'")
        da_reference, da_prediction = mlverif_stats.add_persistence_to_dataarray(
            da_reference, da_prediction
        )
    # Determine what the time dimension to preserve is called
    time_axis_name = "time"
    if time_axis == "elapsed":
        time_axis_name = "elapsed_forecast_duration"

    time_axis_items = time_axis.split(".")
    groupby: str | None = None
    group: str | int | None = None
    len_time_axis_items = len(time_axis_items)
    if 2 <= len_time_axis_items <= 3:
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
    elif len_time_axis_items > 3:
        raise ValueError(
            "Expected 'time_axis' to be in format 'groupedby.{{grouping}}."
            f"{{group}}', got {time_axis}"
        )

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
        da_metric = time_operation(
            da_metric, **time_op_kwargs if time_op_kwargs is not None else {}
        )

    if group is not None:
        if isinstance(da_metric, xr.DataArray):
            da_metric: xr.DataArray = da_metric.sel({groupby: group})
        elif isinstance(da_metric, xr.core.groupby.DataArrayGroupBy):
            da_metric: xr.DataArray = da_metric[group]
        else:
            raise ValueError(
                "da_metric must be an xr.DataArray or xr.core.groupby.DataArrayGroupBy"
            )
    elif isinstance(da_metric, xr.core.groupby.DataArrayGroupBy):
        are_groups_one_dimensional = all(
            len(da_metric[i]) == 1 for i in range(len(da_metric))
        )
        # If all groups are one-dimensional, apply average to turn da_metric
        # into an xr.DataArray
        if are_groups_one_dimensional:
            da_metric: xr.DataArray = da_metric.mean()
        else:
            raise ValueError(
                "`time_operation` must be provided to reduce the "
                "da_metric from an xr.core.groupby.DataArrayGroupBy object to "
                "an xr.DataArray before plotting."
            )
    if axes is None:
        _, axes = plt.subplots()
    num_expected_dims = 1 + (hue in da_metric.dims)
    if len(da_metric.dims) != num_expected_dims:
        raise ValueError(
            f"Metric DataArray must have {num_expected_dims} dimension to plot a timeseries with hue {hue}. da_metric, however, has dims {da_metric.dims}"
        )
    if hue not in da_metric.coords:
        raise ValueError(
            f"DataArray does not contain a coordinate named {hue}, "
            + "please use a different coordinate as hue"
        )

    # To be able to plot "elapsed" time axis together with `preserve_dim`
    # dimension, we need to convert time axis to a format that supports
    # DTypePromotion to that of `preserve_dim`
    if time_axis == "elapsed":
        time_dtype = da_metric[time_axis_name].dtype
        da_metric = da_metric.assign_coords(
            {
                time_axis_name: da_metric[time_axis_name].astype(int),
            }
        )
    if xarray_plot_kwargs and "add_legend" not in xarray_plot_kwargs:
        xarray_plot_kwargs["add_legend"] = da_metric[hue].count().values != 1

    da_metric.plot.line(
        ax=axes,
        hue=hue,
        **xarray_plot_kwargs if xarray_plot_kwargs is not None else {},
    )

    # Set the x-axis ticks to match original time dtype
    if time_axis == "elapsed":
        axes.set_xticks(
            da_metric[time_axis_name].values,
            da_metric[time_axis_name].values.astype(time_dtype).astype(str),
            rotation=45,
            ha="right",
        )

    return axes


def plot_single_metric_gridded_map(
    da_reference: xr.DataArray,
    da_prediction: xr.DataArray,
    stats_operation: Callable = mlverif_stats.difference,
    axes: Optional[plt.Axes] = None,
    time_selection: Optional[Literal["groupedby.{grouping}.{group}"] | datetime] = None,
    time_operation: Optional[Callable] = None,
    time_op_kwargs: Optional[dict] = None,
    xarray_plot_kwargs: Optional[dict] = None,
):
    """Plot a single-metric-gridded-map diagram for a given metric.

    The metric is calculated from da_reference and da_prediction, which should
    have one of the following specification:

    Dimensions: [time, x, y]
    Data variables:
    - state [time, x, y]:
    Coordinates:
    - time:
        the analysis time as a datetime object
    - x:
        the x coordinate of the gridpoint
    - y:
        the y coordinate of the gridpoint

    The `time` dimension can be omitted from the dataarray. If it is present,
    one has to either provide a `time_selection`, and/or a `time_operation`
    callable and possibly a `time_op_kwargs` dictionary with kwargs to pass
    to the callable, to reduce the dimensions of the dataarray before plotting.
    This could e.g. be mlverif_stats.mean, in the case where one wants to
    average the calculated metric over time.

    Parameters
    ----------
    da_reference : xr.DataArray
        Reference dataarray.
    da_prediction : xr.DataArray
        Prediction dataarray.
    stats_operation : Callable, optional
        Statistics operation to calculate the metric, by default mlverif_stats.rmse
    axes : plt.Axes, optional
        Axes to plot on, by default None
    time_selection : Optional[Literal["groupedby.{grouping}.{group}"] | datetime], optional
        The time selection to plot, by default None
    time_operation : Optional[Callable], optional
        Time operation to apply to the time dimension of the calculate da_metric
        dataarray before plotting, by default None
    time_op_kwargs : Optional[dict], optional
        kwargs to pass to the time operation, by default None
    xarray_plot_kwargs : dict, optional
        Additional arguments to pass to xarray's plot function, by default None

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
        da_prediction = da_prediction.sel(time=time_selection)
    elif isinstance(time_selection, str):
        time_selection_items = time_selection.split(".")
        if len(time_selection_items) < 3:
            raise ValueError(
                "Expected 'time_selection' to have format "
                f"'groupedby.{{grouping}}.{{group}}', got {time_selection}"
            )
        if time_selection_items[0] != "groupedby":
            raise ValueError(
                f"Expected 'time_selection' to start with 'groupedby', got {time_selection}"
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
        da_metric = time_operation(
            da_metric, **time_op_kwargs if time_op_kwargs is not None else {}
        )

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
                "`time_operation` must be provided to reduce dimensionality of "
                "the dataarray `da_metric` before plotting."
            )
        raise ValueError(
            "Please select a specific time to plot with 'time_selection=<datetime>'"
        )
    if len(da_metric.dims) != 2:
        raise ValueError(
            "Metric DataArray must have 2 dimensions (x, y) to plot a gridded map"
        )
    da_metric.plot.pcolormesh(
        x="lon",
        y="lat",
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
    time_axis: Literal["groupedby.{grouping}.{group}", "elapsed", "UTC"] = "elapsed",
    time_operation: Optional[Callable] = None,
    time_op_kwargs: Optional[dict] = None,
    xarray_plot_kwargs: Optional[dict] = None,
):
    """Plot a single-metric-hovmoller diagram for a given metric.

    The plot will have time on the x-axis, and the `preserve_dim`
    dimension on the y-axis.

    The metric is calculated from da_reference and da_prediction, which should
    have one of the following specifications:

    A) For `time_axis="elapsed"`:

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

    B) For `time_axis="UTC"` or `time_axis="groupedby.{grouping}.{group}"`:

    Dimensions: [time, `spatial_dim`, ...]
    Coordinates:
    - time:
        the time coordinate as a datetime object
    - `spatial_dim`:
        the coordinate of the spatial dimension to be plotted up the y-axis
    - ...:
        Any other dimensions, which will be reduced along

    In case A), the `start_time` dimension can be omitted from the dataarray.
    If it is present, one has to specify a `time_operation` callable,
    and possibly a `time_op_kwargs` dictionary with kwargs to pass to the callable,
    to reduce the dimensions of the dataarray before plotting.

    In case B), if the provided `time_axis` is "groupedby.{grouping}", i.e.
    with no ".{group}" specified, in most cases a `time_operation` callable is
    required to reduce the xr.core.groupby.DataArrayGroupBy to an xr.DataArray.
    Only in the case where all data arrays of the xr.core.groupby.DataArrayGroupBy
    object have length two, this is not needed (the reduction is automatically
    handled by the function).

    Parameters
    ----------
    da_reference : xr.DataArray
        Reference dataarray.
    da_prediction : xr.DataArray
        Prediction dataarray.
    preserve_dim : str
        Dimension to preserve along the y-axis.
    stats_operation : Callable, optional
        Statistics operation to calculate the metric, by default mlverif_stats.rmse
    axes : plt.Axes, optional
        Axes to plot on, by default None
    time_axis : Literal["groupedby.{grouping}.{group}", "elapsed", "UTC"], optional
        Time axis to use when plotting, by default "elapsed"
    time_operation : Optional[Callable], optional
        Time operation to apply to the time dimension of the calculate da_metric
        dataarray before plotting, by default None
    time_op_kwargs : Optional[dict], optional
        kwargs to pass to the time operation, by default None
    xarray_plot_kwargs : dict, optional
        Additional arguments to pass to xarray's plot function, by default None

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
    # Determine what the time dimension to preserve is called
    if time_axis == "elapsed":
        time_axis_name = "elapsed_forecast_duration"
        time_dim_names = ["start_time", "elapsed_forecast_duration"]
    else:
        time_axis_name = "time"
        time_dim_names = ["time"]

    time_axis_items = time_axis.split(".")
    groupby: str | None = None
    group: str | int | None = None
    len_time_axis_items = len(time_axis_items)
    if 2 <= len_time_axis_items <= 3:
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
        else:
            time_axis_name = groupby
    elif len_time_axis_items > 3:
        raise ValueError(
            "Expected 'time_axis' to be in format 'groupedby.{{grouping}}."
            f"{{group}}', got {time_axis}"
        )

    # Apply statistical operation
    da_metric: xr.DataArray | xr.core.groupby.DataArrayGroupBy = stats_operation(
        da_reference,
        da_prediction,
        groupby=f"time.{groupby}" if groupby is not None else groupby,
        preserve_dims=[
            preserve_dim,
            *time_dim_names,
        ],
    )
    if time_operation is not None:
        da_metric = time_operation(
            da_metric, **time_op_kwargs if time_op_kwargs is not None else {}
        )

    if group is not None:
        if isinstance(da_metric, xr.DataArray):
            da_metric: xr.DataArray = da_metric.sel({groupby: group})
        elif isinstance(da_metric, xr.core.groupby.DataArrayGroupBy):
            da_metric: xr.DataArray = da_metric[group]
        else:
            raise ValueError(
                "da_metric must be an xr.DataArray or xr.core.groupby.DataArrayGroupBy"
            )
    elif isinstance(da_metric, xr.core.groupby.DataArrayGroupBy):
        are_groups_two_dimensional = all(
            len(da_metric[i]) == 2 for i in range(len(da_metric))
        )
        # If all groups are one-dimensional, apply average to turn da_metric
        # into an xr.DataArray
        if are_groups_two_dimensional:
            da_metric: xr.DataArray = da_metric.mean()
        else:
            raise ValueError(
                "`time_operation` must be provided to reduce the "
                "da_metric from an xr.core.groupby.DataArrayGroupBy object to "
                "an xr.DataArray before plotting."
            )

    if axes is None:
        _, axes = plt.subplots()

    # Check if dimensions are present
    if len(da_metric.dims) != 2:
        raise ValueError(
            f"Metric dataarray must have 2 dimensions ({time_axis_name}, "
            f"{preserve_dim}) to plot the gridded map. Got {da_metric.dims}."
        )

    # To be able to plot "elapsed" time axis together with `preserve_dim`
    # dimension, we need to convert time axis to a format that supports
    # DTypePromotion to that of `preserve_dim`
    if time_axis == "elapsed":
        time_dtype = da_metric[time_axis_name].dtype
        da_metric = da_metric.assign_coords(
            {
                time_axis_name: da_metric[time_axis_name].astype(int),
            }
        )

    da_metric.plot.pcolormesh(
        x=time_axis_name,
        y=preserve_dim,
        ax=axes,
        **xarray_plot_kwargs if xarray_plot_kwargs is not None else {},
    )
    # Set the x-axis ticks to match original time dtype
    if time_axis == "elapsed":
        axes.set_xticks(
            da_metric[time_axis_name].values,
            da_metric[time_axis_name].values.astype(time_dtype).astype(str),
            rotation=45,
            ha="right",
        )

    return axes
