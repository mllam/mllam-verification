"""Unit tests for the statistics module."""

import pytest
import xarray as xr

from mllam_verification.operations.statistics import compute_pipeline_statistic, rmse


class TestComputePipelineStatistic:
    """Unit tests for the compute_pipeline_statistic function."""

    def test_compute_pipeline_statistic(self, da_prediction_2d: xr.DataArray):
        """Test computing a pipeline statistic."""
        da_stat = compute_pipeline_statistic(
            [da_prediction_2d],
            stats_op="mean",
            stats_op_kwargs={"dim": ["x", "y"]},
        )
        assert isinstance(da_stat, xr.DataArray)


class TestRmse:
    """Unit tests for the rmse function."""

    def test_rmse(self, da_prediction_2d: xr.DataArray, da_reference_2d: xr.DataArray):
        """Test computing the root mean squared error."""
        da_rmse = rmse(da_prediction_2d, da_reference_2d, reduce_dims=["x", "y"])
        assert isinstance(da_rmse, xr.DataArray)
        assert "cell_methods" in da_rmse.attrs
