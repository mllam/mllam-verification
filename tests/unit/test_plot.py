"""Unit tests for the plot module."""

from datetime import datetime
from typing import Callable, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import pytest
import xarray as xr

import mllam_verification.operations.statistics as mlverif_stats
from mllam_verification.plot import (
    plot_single_metric_gridded_map,
    plot_single_metric_hovmoller,
    plot_single_metric_timeseries,
)


@pytest.fixture(name="time_axis_parameters")
def fixture_time_type_parameters(
    request,
    da_reference_2d_utc,
    da_prediction_2d_utc,
    da_reference_2d_elapsed,
    da_prediction_2d_elapsed,
) -> Tuple[xr.DataArray, xr.DataArray, Callable, bool, str, Callable, dict, int]:
    """Return a tuple of parameters for the test plot functions."""
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


class TestPlotSingleMetricTimeseries:
    """Unit tests for the plot_single_metric_gridded_map function"""

    @pytest.mark.parametrize(
        "time_axis_parameters",
        [
            (
                mlverif_stats.mae,
                False,
                "elapsed",
                mlverif_stats.mean,
                {"dim": "start_time"},
                1,
            ),
            (
                mlverif_stats.rmse,
                False,
                "elapsed",
                mlverif_stats.mean,
                {"dim": "start_time"},
                1,
            ),
            (mlverif_stats.mae, False, "groupedby.hour.0", None, {}, 1),
            (mlverif_stats.rmse, False, "groupedby.hour", mlverif_stats.mean, {}, 1),
        ],
        indirect=True,
    )
    def test_expected_input(
        self,
        time_axis_parameters,
    ):
        """Test with expected input arguments."""
        (
            da_reference,
            da_prediction,
            stats_operation,
            include_persistence,
            time_axis,
            time_operation,
            time_op_kwargs,
            expected_num_lines,
        ) = time_axis_parameters
        ax = plot_single_metric_timeseries(
            da_reference,
            da_prediction,
            stats_operation=stats_operation,
            include_persistence=include_persistence,
            time_axis=time_axis,
            time_operation=time_operation,
            time_op_kwargs=time_op_kwargs,
        )
        assert isinstance(ax, plt.Axes)
        assert len(ax.lines) == expected_num_lines

    @pytest.mark.parametrize(
        "time_axis_parameters",
        [
            (
                mlverif_stats.mae,
                True,
                "elapsed",
                mlverif_stats.mean,
                {"dim": "start_time"},
                1,
            ),
            (
                mlverif_stats.rmse,
                True,
                "elapsed",
                None,
                {"dim": "start_time"},
                1,
            ),
            (mlverif_stats.mae, False, "grouped.hour.0", None, {}, 1),
            (mlverif_stats.rmse, False, "groupedby.hour", None, {}, 1),
            (mlverif_stats.rmse, False, "groupedby.time.hour.0", None, {}, 1),
        ],
        indirect=True,
    )
    def test_unexpected_input(
        self,
        time_axis_parameters,
    ):
        """Test with wrong input arguments."""
        (
            da_reference,
            da_prediction,
            stats_operation,
            include_persistence,
            time_axis,
            time_operation,
            time_op_kwargs,
            _,
        ) = time_axis_parameters
        with pytest.raises(ValueError):
            plot_single_metric_timeseries(
                da_reference,
                da_prediction,
                stats_operation=stats_operation,
                include_persistence=include_persistence,
                time_axis=time_axis,
                time_operation=time_operation,
                time_op_kwargs=time_op_kwargs,
            )


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
        """Test with expected input arguments."""
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
        """Test with wrong input arguments.

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
            plot_single_metric_gridded_map(
                da_reference_2d_utc.isel(datasource=0).drop_vars("datasource"),
                da_prediction_2d_utc.isel(datasource=0).drop_vars("datasource"),
                time_selection=time_selection,
                time_operation=time_operation,
            )


class TestPlotSingleMetricHovmoller:
    """Unit tests for the plot_single_metric_hovmoller function"""

    @pytest.mark.parametrize(
        "time_axis_parameters",
        [
            (
                mlverif_stats.mae,
                None,
                "elapsed",
                mlverif_stats.mean,
                {"dim": "start_time"},
                None,
            ),
            (mlverif_stats.mae, None, "UTC", None, {}, None),
            (mlverif_stats.mae, None, "groupedby.hour", mlverif_stats.mean, {}, None),
            (mlverif_stats.rmse, None, "groupedby.hour.0", None, {}, None),
        ],
        indirect=True,
    )
    def test_expected_input(
        self,
        time_axis_parameters,
    ):
        """Test with expected input arguments."""

        (
            da_reference,
            da_prediction,
            stats_operation,
            _,
            time_axis,
            time_operation,
            time_op_kwargs,
            _,
        ) = time_axis_parameters
        ax = plot_single_metric_hovmoller(
            da_reference.isel(datasource=0),
            da_prediction.isel(datasource=0),
            preserve_dim="y",
            stats_operation=stats_operation,
            time_axis=time_axis,
            time_operation=time_operation,
            time_op_kwargs=time_op_kwargs,
        )
        assert isinstance(ax, plt.Axes)

    @pytest.mark.parametrize(
        "time_axis_parameters",
        [
            (
                mlverif_stats.mae,
                None,
                "elapsed",
                None,
                {},
                None,
            ),
            (mlverif_stats.mae, None, "groupedby.hour", None, {}, None),
            (
                mlverif_stats.rmse,
                None,
                "groupedby.hour.0",
                mlverif_stats.mean,
                {},
                None,
            ),
        ],
        indirect=True,
    )
    def test_unexpected_input(
        self,
        time_axis_parameters,
    ):
        """Test with wrong input arguments."""

        (
            da_reference,
            da_prediction,
            stats_operation,
            _,
            time_axis,
            time_operation,
            time_op_kwargs,
            _,
        ) = time_axis_parameters
        with pytest.raises(ValueError):
            plot_single_metric_hovmoller(
                da_reference.isel(datasource=0),
                da_prediction.isel(datasource=0),
                preserve_dim="y",
                stats_operation=stats_operation,
                time_axis=time_axis,
                time_operation=time_operation,
                time_op_kwargs=time_op_kwargs,
            )
