import matplotlib.pyplot as plt
import xarray as xr
import mllam_verification as mlverif
 
 
def plot_single_metric_timeseries(
    ds_reference: xr.Dataset,
    ds_prediction: xr.Dataset,
    variable: str,
    metric: str,
    axes: plt.Axes = None,
    include_persistence=True,
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
 
    ds_metric = mlverif.calculate_{metric}(ds_reference, ds_prediction, variable, include_persistence=include_persistence)
 
    if axes is None:
        axes = mlverif.operations.plot_utils.get_axes(plot_type="timeseries")
 
    ds_metric[metric].plot.line(ax=axes, **xarray_plot_kwargs)
 
    if include_persistence:
        ds_metric["persistence"].plot.line(ax=axes, **xarray_plot_kwargs)
 
 
    return axes