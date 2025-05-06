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


@pytest.fixture(name="time_type_parameters")
def fixture_time_type_parameters(
    request,
    da_reference_2d_utc,
    da_prediction_2d_utc,
    da_reference_2d_elapsed,
    da_prediction_2d_elapsed,
) -> tuple:
    stats_operation, include_persistence, time_type, groupby, expected_num_lines = (
        request.param
    )
    if time_type == "elapsed":
        return (
            da_reference_2d_elapsed,
            da_prediction_2d_elapsed,
            stats_operation,
            include_persistence,
            time_type,
            groupby,
            expected_num_lines,
        )
    elif time_type == "grouped":
        return (
            da_reference_2d_utc,
            da_prediction_2d_utc,
            stats_operation,
            include_persistence,
            time_type,
            groupby,
            expected_num_lines,
        )
    raise ValueError(
        f"Unknown time_type: {time_type}. Cannot determine which reference array fixture to use."
    )


@pytest.mark.parametrize(
    "time_type_parameters",
    [
        (mlverif_stats.mae, True, "elapsed", None, 2),
        (mlverif_stats.mae, False, "grouped", "time.hour", 1),
        (mlverif_stats.rmse, True, "elapsed", None, 2),
        (mlverif_stats.rmse, False, "grouped", "time.hour", 1),
    ],
    indirect=True,
)
def test_plot_single_metric_timeseries(
    time_type_parameters,
):
    """Test plotting a single-metric-timeseries diagram with different parameters."""

    (
        da_reference,
        da_prediction,
        stats_operation,
        include_persistence,
        time_type,
        groupby,
        expected_num_lines,
    ) = time_type_parameters
    ax = plot_single_metric_timeseries(
        da_reference,
        da_prediction,
        stats_operation=stats_operation,
        include_persistence=include_persistence,
        time_type=time_type,
        groupby=groupby,
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

    plt.show()


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
