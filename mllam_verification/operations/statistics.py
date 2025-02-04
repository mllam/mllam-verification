from typing import List

import scores as scc
import xarray as xr


def rmse(
    ds_prediction: xr.Dataset, ds_reference: xr.Dataset, reduce_dims: List["str"]
) -> xr.Dataset:
    """Compute the root mean squared error across grid_index for all variables.

    Args:
        ds (xr.Dataset): Input dataset
    Returns:
        xr.Dataset: Dataset with the computed statistical variables
    """
    ds_rmse = scc.rmse(ds_prediction, ds_reference, reduce_dims=reduce_dims)

    # Update cell_methods attributes
    for _, da_var in ds_rmse.items():
        da_var.attrs["cell_methods"] = ",".join(reduce_dims) + ": root_mean_square"

    return ds_rmse
