from types import FunctionType
from typing import List, Optional

import scores.continuous as scc_cont
import xarray as xr


def compute_pipeline_statistic(
    datasets: List[xr.Dataset],
    stats_op: Optional[str | FunctionType] = None,
    stats_op_kwargs: Optional[dict] = {},
    diff_dim: Optional[str] = None,
    n_diff_steps: Optional[int] = 1,
    groupby: Optional[str] = None,
) -> xr.Dataset:
    """Apply a series of operations to compute a specific compound statistic.

    The operations applied in order are:
    1. (If diff_dim != None) Apply diff over the `diff_dim` dimension (default to 1 step diff)
    2. (If groupby != None) Apply grouping of dataset according to the `groupby` index
    3. Apply the stats_op to the dataarray with the stats_op_kwargs.

    Parameters
    ----------
    datasets : List[xr.Dataset]
        Datasets to compute the statistic on
    stats_op : str | FunctionType, optional
        Statistic operation to apply. If a string, it must be a valid xarray
        operation. If a FunctionType, it must be a function that takes in
        the datasets and returns one new dataset, by default None
    stats_op_kwargs : dict, optional
        Keyword arguments to pass to the stats_op function, by default {}
    diff_dim : str, optional
        Dimension to apply diff over, by default None
    n_diff_steps : int, optional
        Number of steps to compute the diff over, by default 1
    groupby : str, optional
        Index to group over, by default None
    """
    # Build up CF compliant cell-method attribute so that people know what
    # operations were applied
    cell_methods = []

    for ds in datasets:
        if diff_dim:
            # Only keep variables that have the diff_dim as a dimension
            vars_to_keep = [v for v in ds.data_vars if diff_dim in ds[v].dims]
            if not vars_to_keep:
                raise ValueError(f"No variables found with dimension {diff_dim}")

            # Apply the diff operation
            ds = ds[vars_to_keep].diff(dim=diff_dim, n=n_diff_steps)
            # Get unit of the diff'ed array
            diff_unit_array: xr.DataArray = ds[diff_dim][1] - ds[diff_dim][0]
            diff_unit = diff_unit_array.values
            if diff_dim == "elapsed_forecast_duration":
                # Convert the diff unit to hours
                diff_unit = diff_unit.astype("timedelta64[h]")
            else:
                raise NotImplementedError(
                    f"diff_dim of type {type(diff_dim)} not supported"
                )

            # Update the cell_methods with the operation applied
            cell_methods.append(f"{diff_dim}: diff (interval: {diff_unit})")

        if groupby:
            # Apply the groupby operation
            ds = ds.groupby(groupby)
            # Update the cell_methods with the operation applied
            cell_methods.append(f"{groupby}: groupby")

    if stats_op:
        if isinstance(stats_op, FunctionType):
            ds_stat: xr.Dataset = stats_op(*datasets, **stats_op_kwargs)
        elif isinstance(stats_op, str):
            if len(datasets) == 1:
                # Assume that the stats_op is an xarray operation
                ds_stat: xr.Dataset = getattr(datasets[0], stats_op)(**stats_op_kwargs)
            else:
                raise ValueError(
                    "stats_op as a string is only supported for a single dataset"
                )
        else:
            raise NotImplementedError(f"stats_op {stats_op} not supported")

    cell_methods_str = " ".join(cell_methods)
    # Add cell_methods attribute to all variables
    for var in ds_stat.data_vars:
        ds_stat[var].attrs["cell_methods"] = cell_methods_str

    return ds_stat


def rmse(
    ds_prediction: xr.Dataset, ds_reference: xr.Dataset, reduce_dims: List[str]
) -> xr.Dataset:
    """Compute the root mean squared error across grid_index for all variables.

    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    ds_rmse = compute_pipeline_statistic(
        datasets=[ds_reference, ds_prediction],
        stats_op=scc_cont.rmse,
        stats_op_kwargs={"reduce_dims": reduce_dims},
    )

    # Update cell_methods attributes
    for _, da_var in ds_rmse.items():
        da_var.attrs["cell_methods"] = ",".join(reduce_dims) + ": root_mean_square"

    return ds_rmse


def mae(
    ds_prediction: xr.Dataset, ds_reference: xr.Dataset, reduce_dims: List[str]
) -> xr.Dataset:
    """Compute the mean absolute error across specified dimensions.

    Args:
        ds_prediction (xr.Dataset): Prediction dataset
        ds_reference (xr.Dataset): Reference dataset
        reduce_dims (List[str]): Dimensions to reduce over
    Returns:
        xr.Dataset: Dataset with the mean absolute error computed
    """
    ds_rmse = compute_pipeline_statistic(
        datasets=[ds_reference, ds_prediction],
        stats_op=scc_cont.mae,
        stats_op_kwargs={"reduce_dims": reduce_dims},
    )

    # Update cell_methods attributes
    for _, da_var in ds_rmse.items():
        da_var.attrs["cell_methods"] = ",".join(reduce_dims) + ": mean_absolute_error"

    return ds_rmse


def mean(ds: xr.Dataset, reduce_dims: List[str]) -> xr.Dataset:
    """Compute the mean across specified dimensions.

    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the mean computed
    """
    ds_mean = compute_pipeline_statistic(
        datasets=[ds],
        stats_op="mean",
        stats_op_kwargs={"dim": reduce_dims},
    )

    # Update cell_methods attributes
    for _, da_var in ds_mean.items():
        da_var.attrs["cell_methods"] = ",".join(reduce_dims) + ": mean"

    return ds_mean
