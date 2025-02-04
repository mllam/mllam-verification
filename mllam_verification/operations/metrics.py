from typing import List

import xarray as xr

from .statistics import rmse


def calculate_rmse(
    ds_reference: xr.Dataset,
    ds_prediction: xr.Dataset,
    variable: str,
    reduce_dims: List[str],
    include_persistence=False,
) -> xr.Dataset:
    """Calculate RMSE between prediction and reference datasets.

    RMSE: Root Mean Square Error

    If specified, calculate the error relative to persistence too.
    The calculation is done only for the specified variable.
    The input datasets are assumed to have the following specifications:

    Dimensions: [start_time, elapsed_forecast_duration, reduce_dim1, reduce_dims2, ...]
    Data variables:
    - variable1 [start_time, elapsed_forecast_duration, reduce_dim1, reduce_dims2, ...]:
    - variable2 [start_time, elapsed_forecast_duration, reduce_dim1, reduce_dims2, ...]:
    - ...
    - variableN [start_time, elapsed_forecast_duration, reduce_dim1, reduce_dims2, ...]:
    Coordinates:
    - start_time:
        the analysis time as a datetime object
    - elapsed_forecast_duration:
        the elapsed forecast duration as a timedelta object
    - reduce_dim1:
        one of the dimensions to reduce along when calculating the persistence
    - reduce_dim2:
        one of the dimensions to reduce along when calculating the persistence
    - ...

    The error is averaged along the start_time dimension of the datasets.
    The error is returned as a dataset with the following specification:

    Dimensions: [elapsed_forecast_duration]
    Data variables:
    - <variable>_rmse [elapsed_forecast_duration]:
        the RMSE between the prediction and reference datasets
    - <variable>_persistence [elapsed_forecast_duration], optional:
        the persistence RMSE calculated based on the reference datasets
    Coordinates:
    - elapsed_forecast_duration:

    Parameters:
    -----------
    ds_reference: xr.Dataset
        The reference dataset to calculate global error against.
    ds_prediction: xr.Dataset
        The prediction dataset to calculate global error of.
    variable: str
        The variable to calculate the metric of.
    reduce_dims: List[str]
        The dimensions to reduce along when calculating the metric.
    include_persistence: bool
        Whether to calculate the error relative to persistence
    """
    # Select the variable from the datasets
    ds_reference = ds_reference[[variable]]
    ds_prediction = ds_prediction[[variable]]

    # Calculate the error and rename the variable
    ds_metric = rmse(ds_prediction, ds_reference, reduce_dims=reduce_dims)
    ds_metric = ds_metric.rename({variable: f"{variable}_rmse"})

    # Calculate the persistence error and merge with the metric dataset
    if include_persistence:
        ds_persistence_metric = calculate_persistence_rmse(
            ds_reference, variable, reduce_dims=reduce_dims
        )
        ds_metric = xr.merge([ds_metric, ds_persistence_metric])

    # Take mean over all start times
    ds_metric = ds_metric.mean("start_time")
    # Update cell_methods attributes
    for _, da_var in ds_metric.items():
        da_var.attrs["cell_methods"] = " ".join(
            [da_var.attrs["cell_methods"], "start_time: mean"]
        )

    return ds_metric


def calculate_persistence_rmse(
    ds_reference: xr.Dataset, variable: str, reduce_dims: List[str]
) -> xr.Dataset:
    """Calculate the RMSE between the reference dataset and its persistence.

    RMSE: Root Mean Square Error

    The calculation is done only for the specified variable.
    The input dataset is assumed to have the following specifications:

    Dimensions: [start_time, elapsed_forecast_duration, reduce_dim1, reduce_dims2, ...]
    Data variables:
    - variable1 [start_time, elapsed_forecast_duration, reduce_dim1, reduce_dims2, ...]:
    - variable2 [start_time, elapsed_forecast_duration, reduce_dim1, reduce_dims2, ...]:
    - ...
    - variableN [start_time, elapsed_forecast_duration, reduce_dim1, reduce_dims2, ...]:
    Coordinates:
    - start_time:
        the analysis time as a datetime object
    - elapsed_forecast_duration:
        the elapsed forecast duration as a timedelta object
    - reduce_dim1:
        one of the dimensions to reduce along when calculating the persistence
    - reduce_dim2:
        one of the dimensions to reduce along when calculating the persistence
    - ...

    The error is returned as a dataset with the following specification:

    Dimensions: [elapsed_forecast_duration]
    Data variables:
    - <variable>_persistence [elapsed_forecast_duration]:
        the persistence RMSE calculated based on the reference dataset
    Coordinates:
    - elapsed_forecast_duration:

    Parameters:
    -----------
    ds_reference: xr.Dataset
        The reference dataset to calculate persistence error against.
    variable: str
        The variable to calculate the persistence of.
    reduce_dims: List[str]
        The dimensions to reduce along when calculating the persistence.
    """
    # Select the variable from the dataset
    ds_reference = ds_reference[[variable]]

    ds_persistence_reference = ds_reference.isel(
        elapsed_forecast_duration=slice(1, None)
    )
    ds_persistence_prediction = ds_reference.isel(
        elapsed_forecast_duration=slice(0, -1)
    )
    ds_persistence_prediction = ds_persistence_prediction.assign_coords(
        elapsed_forecast_duration=ds_persistence_reference["elapsed_forecast_duration"]
    )
    ds_persistence_metric = rmse(
        ds_prediction=ds_persistence_prediction,
        ds_reference=ds_persistence_reference,
        reduce_dims=reduce_dims,
    )
    ds_persistence_metric = ds_persistence_metric.rename(
        {variable: f"{variable}_persistence"}
    )

    return ds_persistence_metric
