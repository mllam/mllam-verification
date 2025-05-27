"""Module for validating function arguments."""

import re
from typing import Optional

import xarray as xr


def validate_time_axis(value: str) -> str:
    """Validator for the time_axis argument

    Parameters
    ----------
    value : str
        The value to validate

    Raises
    ------
    ValueError
        If the value is not a valid time axis

    Returns
    -------
    str
        The validated value
    """
    # Define the regex pattern for matching "groupedby.{grouping}.{group}"
    pattern = (
        r"^groupedby\.(?P<grouping>[a-zA-Z_][a-zA-Z0-9_]*)\.?(?P<group>[a-zA-Z0-9]*)?$"
    )
    if not re.match(pattern, value):
        raise ValueError(
            "Invalid time_axis/time_selection format. Must be "
            "'groupedby.{grouping}.{group}', 'elapsed', or 'UTC',"
            "where {grouping} and {group} (optional) are strings."
        )
    return value


def validate_time_selection(value: str) -> str:
    """Validator for the time_selection argument

    Parameters
    ----------
    value : str
        The value to validate

    Raises
    ------
    ValueError
        If the value is not a valid time selection

    Returns
    -------
    str
        The validated value
    """
    # Define the regex pattern for matching "groupedby.{grouping}.{group}"
    pattern = (
        r"^groupedby\.(?P<grouping>[a-zA-Z_][a-zA-Z0-9_]*)\.(?P<group>[a-zA-Z0-9]*)$"
    )
    if not re.match(pattern, value):
        raise ValueError(
            "Expected 'time_selection' to be either None, a datetime or a string "
            "with format 'groupedby.{{grouping}}.{{group}}, where {grouping} and"
            "{group} are strings.'"
        )
    return value


def check_dims_for_gridded_map(da: xr.DataArray, groupby: Optional[str] = None):
    """Check if the DataArray has the correct dimensions for plotting gridded map.

    Parameters
    ----------
    da : xr.DataArray
        DataArray to check
    groupby : Optional[str], optional
        Groupby dimension to check, by default None

    Raises
    ------
    ValueError
        If the DataArray does not have the correct dimensions for plotting gridded map
    """
    if "time" in da.dims and da.sizes["time"] >= 1:
        if groupby is not None:
            raise ValueError(
                "`time_operation` must be provided to reduce dimensionality of "
                "the dataarray before plotting."
            )
        raise ValueError(
            "Please select a specific time to plot with 'time_selection=<datetime>'"
        )
    if len(da.dims) != 2:
        raise ValueError(
            "Metric DataArray must have 2 dimensions (x, y) to plot a gridded map"
        )
