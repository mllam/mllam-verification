"""
This module contains pytest fixtures for generating test data for unit tests.
The fixtures create various datasets and arrays used in the tests.
"""

import collections
from typing import Tuple

import numpy as np
import pytest
import xarray as xr
from numpy._typing._array_like import NDArray

DOMAIN_WIDTH = 100
MeshGrid = collections.namedtuple("MeshGrid", "x y")
FORECAST_LENGTH = 5  # hours
FORECAST_SPATIAL_STEP = 15  # in grid points
NFORECASTS = 10


@pytest.fixture(name="start_times", scope="session")
def fixture_start_times():
    """Fixture that returns a numpy array representing start_times."""
    return np.array(
        [
            np.datetime64("today", "h") + np.timedelta64(i * 2 * FORECAST_LENGTH, "h")
            for i in range(NFORECASTS)
        ],
        dtype="datetime64[ns]",
    )


@pytest.fixture(name="elapsed_forecast_duration", scope="session")
def fixture_elapsed_forecast_duration():
    """Fixture that returns a numpy array representing elapsed_forecast_duration steps."""
    return np.array(
        [np.timedelta64(i, "h") for i in range(1, FORECAST_LENGTH)],
        dtype="timedelta64[ns]",
    )


@pytest.fixture(name="reference_times", scope="session")
def fixture_reference_times(start_times: NDArray, elapsed_forecast_duration: NDArray):
    """Fixture that returns a numpy array representing reference_times.

    Reference times consists of the unique values of start_times and
    start_times + elapsed_forecast_duration.
    """
    reference_times = start_times[..., np.newaxis].repeat(
        len(elapsed_forecast_duration) + 1, axis=1
    )
    reference_times[:, 1:] = reference_times[:, 1:] + elapsed_forecast_duration
    return reference_times


@pytest.fixture(name="unique_reference_times", scope="session")
def fixture_unique_reference_times(reference_times: NDArray) -> NDArray:
    """Fixture that returns a numpy array of unique reference times."""
    return np.arange(
        reference_times.min(),
        reference_times.max() + 1,
        step=np.timedelta64(1, "h"),
        dtype="datetime64[ns]",
    )


@pytest.fixture(name="meshgrid", scope="session")
def fixture_meshgrid() -> Tuple[NDArray, NDArray]:
    """Fixture that returns a meshgrid of shape (DOMAIN_WIDTH, DOMAIN_WIDTH)."""
    index = np.arange(0, DOMAIN_WIDTH)
    return MeshGrid(*np.meshgrid(index, index))


@pytest.fixture(name="moving_gaussian_blob", scope="session")
def fixture_moving_gaussian_blob(unique_reference_times: NDArray, meshgrid: MeshGrid):
    """Fixture of a 3D array representing a moving Gaussian blob."""
    gaussian_blob = np.exp(
        -((meshgrid.x - 50) ** 2 + (meshgrid.y - 50) ** 2) / (2 * 10**2)
    )
    return xr.DataArray(
        np.array(
            [
                np.roll(gaussian_blob, shift * FORECAST_SPATIAL_STEP, axis=0)
                for shift, _ in enumerate(unique_reference_times)
            ]
        ),
        coords={"time": unique_reference_times},
        dims=["time", "x", "y"],
    )


@pytest.fixture(name="da_reference_2d_elapsed", scope="session")
def fixture_da_reference_2d_elapsed(
    moving_gaussian_blob: xr.DataArray,
    meshgrid: MeshGrid,
    start_times: NDArray,
    reference_times: NDArray,
    elapsed_forecast_duration: NDArray,
) -> xr.DataArray:
    """Fixture that returns DataArray of 2d moving gaussian blobs reference data.

    The reference data is aligned with the prediction data and only contains
    relevant times.
    """
    da = xr.DataArray(
        [
            moving_gaussian_blob.sel(time=reference_times[forecast, 1:]).values
            for forecast in range(NFORECASTS)
        ],
        coords={
            "start_time": start_times,
            "elapsed_forecast_duration": elapsed_forecast_duration,
            "x": meshgrid.x[0, :],
            "y": meshgrid.y[:, 0],
        },
        dims=["start_time", "elapsed_forecast_duration", "x", "y"],
    )
    return da.assign_coords(datasource="forecast").expand_dims({"datasource": 1}, axis=-1)


@pytest.fixture(name="da_prediction_2d_elapsed", scope="session")
def fixture_da_prediction_2d_elapsed(
    da_reference_2d_elapsed: xr.DataArray,
    start_times: NDArray,
    elapsed_forecast_duration: NDArray,
    meshgrid: MeshGrid,
) -> xr.DataArray:
    """Fixture that returns DataArray of 2d moving gaussian blob prediction data.

    The prediction data is the reference data with added noise and bias.
    """
    bias = 0
    data = da_reference_2d_elapsed.values.copy()
    noise = np.random.normal(0, 0.1, data.shape)
    data += noise + bias
    return xr.DataArray(
        data,
        coords={
            "start_time": start_times,
            "elapsed_forecast_duration": elapsed_forecast_duration,
            "x": meshgrid.x[0, :],
            "y": meshgrid.y[:, 0],
            "datasource": ["forecast"],
        },
    )


@pytest.fixture(name="da_reference_2d_utc", scope="session")
def fixture_da_reference_2d_utc(
    moving_gaussian_blob: xr.DataArray,
    meshgrid: MeshGrid,
) -> xr.DataArray:
    """Fixture that returns DataArray of 2d moving gaussian blobs reference data.

    The reference data is aligned with the prediction data and only contains
    relevant times.
    """
    return moving_gaussian_blob.assign_coords(
        datasource="forecast",
        x=meshgrid.x[0, :],
        y=meshgrid.y[:, 0],
    ).expand_dims({"datasource": 1}, axis=-1)


@pytest.fixture(name="da_prediction_2d_utc", scope="session")
def fixture_da_prediction_2d_utc(
    da_reference_2d_utc: xr.DataArray,
) -> xr.DataArray:
    """Fixture that returns DataArray of 2d moving gaussian blobs reference data.

    The reference data is aligned with the prediction data and only contains
    relevant times.
    """
    bias = 0
    data = da_reference_2d_utc.values.copy()
    noise = np.random.normal(0, 0.1, data.shape)
    data += noise + bias

    return da_reference_2d_utc.copy(data=data)
