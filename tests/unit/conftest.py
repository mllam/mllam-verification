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
NFEATURES = 5
MeshGrid = collections.namedtuple("MeshGrid", "x y")
FORECAST_LENGTH = 5
FORECAST_SPATIAL_STEP = 15
NFORECASTS = 3


@pytest.fixture(name="start_times", scope="session")
def fixture_start_times():
    """Fixture that returns a numpy array representing start_times."""
    return np.arange(0, NFORECASTS * (2 * FORECAST_LENGTH), 2 * FORECAST_LENGTH)


@pytest.fixture(name="elapsed_forecast_duration", scope="session")
def fixture_elapsed_forecast_duration():
    """Fixture that returns a numpy array representing elapsed_forecast_duration steps."""
    return np.arange(1, FORECAST_LENGTH)


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
    return np.arange(reference_times.min(), reference_times.max() + 1)


@pytest.fixture(name="meshgrid", scope="session")
def fixture_meshgrid() -> Tuple[NDArray, NDArray]:
    """Fixture that returns a meshgrid of shape (DOMAIN_WIDTH, DOMAIN_WIDTH)."""
    index = np.arange(0, DOMAIN_WIDTH)
    return MeshGrid(*np.meshgrid(index, index))


@pytest.fixture(name="gaussian_blob", scope="session")
def fixture_gaussian_blob(meshgrid: MeshGrid):
    """Fixture that returns a 2D Gaussian blob.

    The blob is centered at (50, 50) with a standard deviation of 10.
    """
    return np.exp(-((meshgrid.x - 50) ** 2 + (meshgrid.y - 50) ** 2) / (2 * 10**2))


@pytest.fixture(name="moving_gaussian_blob", scope="session")
def fixture_moving_gaussian_blob(
    unique_reference_times: NDArray, gaussian_blob: NDArray
):
    """Fixture of a 3D array representing a moving Gaussian blob."""
    return np.array(
        [
            np.roll(gaussian_blob, shift * FORECAST_SPATIAL_STEP, axis=0)
            for shift in unique_reference_times
        ]
    )


@pytest.fixture(name="moving_gaussian_blobs", scope="session")
def fixture_moving_gaussian_blobs(moving_gaussian_blob: NDArray):
    """Fixture of a 4D array representing multiple moving Gaussian blobs.

    Each blob is shifted by a random dx, dy and signifies an individual
    state_feature.
    """
    return np.array(
        [
            np.roll(moving_gaussian_blob, (dx, dy), axis=(1, 2))
            for dx, dy in np.random.randint(
                -DOMAIN_WIDTH // 2, DOMAIN_WIDTH // 2, size=(NFEATURES, 2)
            )
        ]
    ).transpose((1, 2, 3, 0))


@pytest.fixture(name="ds_reference_1d", scope="session")
def fixture_ds_reference_1d(
    unique_reference_times: NDArray, moving_gaussian_blobs: NDArray
) -> xr.Dataset:
    """Fixture that returns Dataset with 1d moving gaussian blob reference data."""
    data = moving_gaussian_blobs.reshape((len(unique_reference_times), -1, NFEATURES))
    grid_index = np.arange(data.shape[1])
    return xr.Dataset(
        {
            f"feature{i}": (
                [
                    "time",
                    "grid_index",
                ],
                data[..., i],
            )
            for i in np.arange(NFEATURES)
        },
        coords={
            "time": unique_reference_times,
            "grid_index": grid_index,
        },
    )


@pytest.fixture(name="moving_gaussian_blobs_per_reference_time_1d", scope="session")
def fixture_moving_gaussian_blobs_per_reference_time_1d(
    moving_gaussian_blobs: NDArray,
    reference_times: NDArray,
    start_times: NDArray,
    elapsed_forecast_duration: NDArray,
) -> NDArray:
    """Fixture that returns an array of moving gaussian blobs per reference time.

    The dimensions of the array are [NFORECASTS, FORECAST_LENGTH - 1,
    DOMAIN_WIDTH * DOMAIN_WIDTH, NFEATURES].
    """
    reference_times_flattened = reference_times[:, 1:].flatten()
    return moving_gaussian_blobs[reference_times_flattened, ...].reshape(
        len(start_times),
        len(elapsed_forecast_duration),
        -1,
        NFEATURES,
    )


@pytest.fixture(name="ds_reference_1d_relevant_times_and_aligned", scope="session")
def fixture_ds_reference_1d_relevant_times_and_aligned(
    moving_gaussian_blobs_per_reference_time_1d: NDArray,
    start_times: NDArray,
    elapsed_forecast_duration: NDArray,
) -> xr.Dataset:
    """Fixture that returns Dataset 1d moving gaussian blobs reference data.

    The reference data is aligned with the prediction data and only contains
    relevant times.
    """
    grid_index = np.arange(moving_gaussian_blobs_per_reference_time_1d.shape[2])

    return xr.Dataset(
        {
            "state": (
                [
                    "analysis_time",
                    "elapsed_forecast_duration",
                    "grid_index",
                    "state_feature",
                ],
                moving_gaussian_blobs_per_reference_time_1d,
            )
        },
        coords={
            "analysis_time": start_times,
            "elapsed_forecast_duration": elapsed_forecast_duration,
            "grid_index": grid_index,
            "state_feature": [f"feature{i}" for i in np.arange(NFEATURES)],
        },
    )


@pytest.fixture(name="ds_prediction_1d", scope="session")
def fixture_ds_prediction_1d(
    start_times: NDArray,
    elapsed_forecast_duration: NDArray,
    moving_gaussian_blobs_per_reference_time_1d: NDArray,
) -> xr.Dataset:
    """Fixture that returns Dataset with 1d moving gaussian blob prediction data.

    The prediction data is the reference data with added noise and bias.
    """
    data = moving_gaussian_blobs_per_reference_time_1d
    grid_index = np.arange(data.shape[2])

    bias = 0
    noise = np.random.normal(0, 0.1, data.shape)
    data = data + noise + bias

    return xr.Dataset(
        {
            "state": (
                [
                    "analysis_time",
                    "elapsed_forecast_duration",
                    "grid_index",
                    "state_feature",
                ],
                data,
            )
        },
        coords={
            "analysis_time": start_times,
            "elapsed_forecast_duration": elapsed_forecast_duration,
            "grid_index": grid_index,
            "state_feature": [f"feature{i}" for i in np.arange(NFEATURES)],
        },
    )


@pytest.fixture(name="ds_reference_2d", scope="session")
def fixture_ds_reference_2d(
    unique_reference_times: NDArray,
    moving_gaussian_blobs: NDArray,
    meshgrid: MeshGrid,
) -> xr.Dataset:
    """Fixture that returns Dataset 2d moving gaussian blobs reference data."""
    return xr.Dataset(
        {
            f"feature{i}": (
                [
                    "time",
                    "x",
                    "y",
                ],
                moving_gaussian_blobs[..., i],
            )
            for i in np.arange(NFEATURES)
        },
        coords={
            "time": unique_reference_times,
            "x": meshgrid.x[0, :],
            "y": meshgrid.y[:, 0],
        },
    )


@pytest.fixture(name="moving_gaussian_blobs_per_reference_time_2d", scope="session")
def fixture_moving_gaussian_blobs_per_reference_time_2d(
    moving_gaussian_blobs: NDArray, reference_times: NDArray
) -> NDArray:
    """Fixture that returns an array of moving gaussian blobs per reference time.

    The dimensions of the array are [NFORECASTS, FORECAST_LENGTH - 1,
    DOMAIN_WIDTH, DOMAIN_WIDTH, NFEATURES].
    """
    return (
        np.array(
            [
                moving_gaussian_blobs[reference_times[forecast, 1:], ...]
                for forecast in range(NFORECASTS)
            ]
        ),
    )[0]


@pytest.fixture(name="ds_reference_2d_relevant_times_and_aligned", scope="session")
def fixture_ds_reference_2d_relevant_times_and_aligned(
    moving_gaussian_blobs_per_reference_time_2d: NDArray,
    meshgrid: MeshGrid,
    start_times: NDArray,
    elapsed_forecast_duration: NDArray,
) -> xr.Dataset:
    """Fixture that returns Dataset 2d moving gaussian blobs reference data.

    The reference data is aligned with the prediction data and only contains
    relevant times.
    """
    return xr.Dataset(
        {
            f"feature{i}": (
                [
                    "start_time",
                    "elapsed_forecast_duration",
                    "x",
                    "y",
                    "datasource",
                ],
                moving_gaussian_blobs_per_reference_time_2d[..., i, np.newaxis],
            )
            for i in np.arange(NFEATURES)
        },
        coords={
            "start_time": start_times,
            "elapsed_forecast_duration": elapsed_forecast_duration,
            "x": meshgrid.x[0, :],
            "y": meshgrid.y[:, 0],
            "datasource": ["forecast"],
        },
    )


@pytest.fixture(name="ds_prediction_2d", scope="session")
def fixture_ds_prediction_2d(
    moving_gaussian_blobs_per_reference_time_2d: NDArray,
    start_times: NDArray,
    elapsed_forecast_duration: NDArray,
    meshgrid: MeshGrid,
) -> xr.Dataset:
    """Fixture that returns Dataset 2d moving gaussian blob prediction data.

    The prediction data is the reference data with added noise and bias.
    """
    bias = 0
    data = np.copy(moving_gaussian_blobs_per_reference_time_2d)
    noise = np.random.normal(0, 0.1, data.shape)
    data += noise + bias

    return xr.Dataset(
        {
            f"feature{i}": (
                [
                    "start_time",
                    "elapsed_forecast_duration",
                    "x",
                    "y",
                    "datasource",
                ],
                data[..., i, np.newaxis],
            )
            for i in np.arange(NFEATURES)
        },
        coords={
            "start_time": start_times,
            "elapsed_forecast_duration": elapsed_forecast_duration,
            "x": meshgrid.x[0, :],
            "y": meshgrid.y[:, 0],
            "datasource": ["forecast"],
        },
    )
