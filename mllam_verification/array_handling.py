"""Validate and check arrays."""

import xarray as xr


def select_group(
    groupby: str,
    group: str | int,
    da_metric: xr.DataArray | xr.core.groupby.DataArrayGroupBy,
) -> xr.DataArray:
    """Select a specific group from a DataArray or DataArrayGroupBy object.

    Args:
        groupby (str): The name of the groupby dimension
        group (str | int): The group to select
        da_metric (xr.DataArray | xr.core.groupby.DataArrayGroupBy):
            The DataArray or DataArrayGroupBy object to select from

    Raises:
        ValueError: If da_metric is not an xr.DataArray or
            xr.core.groupby.DataArrayGroupBy

    Returns:
        xr.DataArray: The selected DataArray
    """
    if isinstance(da_metric, xr.DataArray):
        da_metric = da_metric.sel({groupby: group})
    elif isinstance(da_metric, xr.core.groupby.DataArrayGroupBy):
        da_metric = da_metric[group]
    else:
        raise ValueError(
            "da_metric must be an xr.DataArray or xr.core.groupby.DataArrayGroupBy"
        )

    return da_metric


def reduce_groups(
    da_metric: xr.core.groupby.DataArrayGroupBy, expected_num_dims: int
) -> xr.DataArray:
    """Reduce xr.core.groupby.DataArrayGroupBy object to an xr.DataArray.

    Only executed if all groups are `expected_num_dims`-dimensional. Otherwise,
    raise an error.

    Args:
        da_metric (xr.core.groupby.DataArrayGroupBy):
            The DataArrayGroupBy object to reduce
        expected_num_dims (int): The expected number of dimensions of each group

    Raises:
        ValueError: If all groups are not `expected_num_dims`-dimensional

    Returns:
        xr.DataArray: The resulting DataArray with reduced groups.
    """

    are_groups_n_dimensional = all(
        len(da_metric[i]) == expected_num_dims for i in range(len(da_metric))
    )
    # If all groups are two-dimensional, apply average to turn da_metric
    # into an xr.DataArray
    if are_groups_n_dimensional:
        return da_metric.mean()
    raise ValueError(
        "`time_operation` must be provided to reduce the "
        "da_metric from an xr.core.groupby.DataArrayGroupBy object to "
        "an xr.DataArray before plotting."
    )
