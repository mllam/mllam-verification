"""Unit tests for the plot module."""

from datetime import datetime
from typing import Callable, Literal, Optional

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
    (
        stats_operation,
        include_persistence,
        time_axis,
        time_operation,
        time_op_kwargs,
        expected_num_lines,
    ) = request.param
    if time_axis == "elapsed":
        return (
            da_reference_2d_elapsed,
            da_prediction_2d_elapsed,
            stats_operation,
            include_persistence,
            time_axis,
            time_operation,
            time_op_kwargs,
            expected_num_lines,
        )
    if "groupedby" in time_axis:
        return (
            da_reference_2d_utc,
            da_prediction_2d_utc,
            stats_operation,
            include_persistence,
            time_axis,
            time_operation,
            time_op_kwargs,
            expected_num_lines,
        )
    raise ValueError(
        f"Unknown time_type: {time_axis}. Cannot determine which reference array fixture to use."
    )


@pytest.mark.parametrize(
    "time_type_parameters",
    [
        (
            mlverif_stats.mae,
            True,
            "elapsed",
            mlverif_stats.mean,
            {"dim": "start_time"},
            2,
        ),
        (
            mlverif_stats.rmse,
            True,
            "elapsed",
            mlverif_stats.mean,
            {"dim": "start_time"},
            2,
        ),
        (mlverif_stats.mae, False, "groupedby.hour.0", None, {}, 1),
        (mlverif_stats.rmse, False, "groupedby.hour", mlverif_stats.mean, {}, 1),
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
        time_operation,
        time_op_kwargs,
        expected_num_lines,
    ) = time_type_parameters
    ax = plot_single_metric_timeseries(
        da_reference,
        da_prediction,
        stats_operation=stats_operation,
        include_persistence=include_persistence,
        time_axis=time_type,
        time_operation=time_operation,
        time_op_kwargs=time_op_kwargs,
    )
    assert isinstance(ax, plt.Axes)
    assert len(ax.lines) == expected_num_lines


class TestPlotSingleMetricGriddedMap:
    """Unit tests for the plot_single_metric_gridded_map function"""

    @pytest.mark.parametrize(
        "time_selection,time_operation",
        [
            (None, None),
            (datetime.today().replace(hour=0, minute=0, second=0, microsecond=0), None),
            ("groupedby.hour.0", mlverif_stats.mean),
        ],
    )
    def test_expected_input(
        self,
        da_reference_2d_utc: xr.DataArray,
        da_prediction_2d_utc: xr.DataArray,
        time_selection: Optional[Literal["groupedby.{grouping}.{group}"] | datetime],
        time_operation: Optional[Callable],
    ):
        """Test plotting a single-metric-timeseries diagram with different parameters."""
        # When time selectin is None, we assume that the user has selected the time
        # to plot themselves. This is simulated by selecting the first time step.
        if time_selection is None:
            da_reference_2d_utc = da_prediction_2d_utc.isel(time=0)
            da_prediction_2d_utc = da_prediction_2d_utc.isel(time=0)

        ax = plot_single_metric_gridded_map(
            da_reference_2d_utc.isel(datasource=0).drop_vars("datasource"),
            da_prediction_2d_utc.isel(datasource=0).drop_vars("datasource"),
            time_selection=time_selection,
            time_operation=time_operation,
        )
        assert isinstance(ax, plt.Axes)

    @pytest.mark.parametrize(
        "time_selection,time_operation,exception",
        [
            (None, None, ValueError),
            (datetime.today(), None, KeyError),
            ("groupedby.hour", None, ValueError),
            ("groupedby.hour.0", None, ValueError),
        ],
    )
    def test_unexpected_input(
        self,
        da_reference_2d_utc: xr.DataArray,
        da_prediction_2d_utc: xr.DataArray,
        time_selection: Optional[Literal["groupedby.{grouping}.{group}"] | datetime],
        time_operation: Optional[Callable],
        exception: Exception,
    ):
        """Test plotting a single-metric-gridded-map diagram with wrong input parameters.

        Args:
            da_reference_2d_utc (xr.DataArray): The reference dataset
            da_prediction_2d_utc (xr.DataArray): The prediction dataset
            time_selection (Optional[Literal["groupedby.{grouping}.{group}"]] | datetime):
                The time selection to plot
            time_operation (Optional[Callable]):
                A time operation to apply to the time selection
            exception (Exception): The expected exception
        """
        with pytest.raises(exception):
            ax = plot_single_metric_gridded_map(
                da_reference_2d_utc.isel(datasource=0).drop_vars("datasource"),
                da_prediction_2d_utc.isel(datasource=0).drop_vars("datasource"),
                time_selection=time_selection,
                time_operation=time_operation,
            )
            assert isinstance(ax, plt.Axes)


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
