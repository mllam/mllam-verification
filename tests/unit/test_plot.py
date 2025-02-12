"""Unit tests for the plot module."""

import matplotlib.pyplot as plt
import pytest
import xarray as xr

import mllam_verification.operations.statistics as mlverif_stats
from mllam_verification.plot import (
    plot_single_metric_gridded_map,
    plot_single_metric_hovmoller,
    plot_single_metric_timeseries,
)

# class TestPlotSingleMetricTimeseries:
#     """Unit tests for the plot_single_metric_timeseries function."""


@pytest.mark.parametrize(
    "stats_operation, include_persistence, expected_num_lines",
    [
        (mlverif_stats.rmse, False, 1),
        (mlverif_stats.mae, True, 2),
    ],
)
def test_plot_single_metric_timeseries(
    da_reference_2d: xr.DataArray,
    da_prediction_2d: xr.DataArray,
    stats_operation,
    include_persistence,
    expected_num_lines,
):
    """Test plotting a single-metric-timeseries diagram with different parameters."""
    ax = plot_single_metric_timeseries(
        da_reference_2d,
        da_prediction_2d,
        stats_operation=stats_operation,
        include_persistence=include_persistence,
    )
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == expected_num_lines


# class TestPlotSingleMetricGriddedMap:
#     """Unit tests for the plot_single_metric_timeseries function."""


# @pytest.mark.parametrize(
#     "stats_operation, include_persistence, expected_num_lines",
#     [
#         (mlverif_stats.rmse, False, 1),
#         (mlverif_stats.mae, True, 2),
#     ],
# )
def test_plot_single_metric_gridded_map(
    da_reference_2d: xr.DataArray,
    da_prediction_2d: xr.DataArray,
):
    """Test plotting a single-metric-timeseries diagram with different parameters."""
    ax = plot_single_metric_gridded_map(
        da_reference_2d.isel(elapsed_forecast_duration=0, datasource=0),
        da_prediction_2d.isel(elapsed_forecast_duration=0, datasource=0),
    )
    assert isinstance(ax, plt.Axes)

    # plt.show()


def test_plot_single_metric_hovmoller(
    da_reference_2d: xr.DataArray,
    da_prediction_2d: xr.DataArray,
):
    """Test plotting a single-metric-timeseries diagram with different parameters."""
    ax = plot_single_metric_hovmoller(
        da_reference_2d.isel(datasource=0),
        da_prediction_2d.isel(datasource=0),
        preserve_dim="y",
    )
    assert isinstance(ax, plt.Axes)

    # plt.show()
