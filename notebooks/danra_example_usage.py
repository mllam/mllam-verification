import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import cartopy.crs as ccrs
import dmidc
import dmidc.harmonie
import dmidc.metobs
import dmidc.utils
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mllam_verification.operations.statistics import mean, rmse
from mllam_verification.plot import (
    plot_single_metric_gridded_map,
    plot_single_metric_timeseries,
)

dmidc.__version__

ANALYSIS_TIME = np.datetime64("2022-01-01")
FORECAST_DURATION = "PT18H"

# Retrieve DANRA forecast data
# with tempfile.TemporaryDirectory() as tempdir:
dir_ = "/home/maf/mllam/data"
# ds_t2m_danra_forecast = dmidc.harmonie.load(
#     suite_name="DANRA",
#     analysis_time=slice("2022-01-01", "2022-01-07"),
#     # slice(ANALYSIS_TIME, ANALYSIS_TIME + np.timedelta64(24, "h")),
#     data_kind="FORECAST",
#     forecast_duration=FORECAST_DURATION,
#     short_name=["t"],
#     level_type="heightAboveGround",
#     level=2,
#     temp_filepath=dir_,
# )
# exit()
fp_zarr_json = "/home/maf/mllam/data/analysis_time_20220101__suite_name_DANRA__data_kind_FC3hr__short_name_t__level_2__level_type_heightAboveGround__forecast_duration_3:00:00.zarr.json"
fp_zarr_json = "/home/maf/mllam/data/analysis_time_20220101-20220107__suite_name_DANRA__data_kind_FC33hr__short_name_t__level_2__level_type_heightAboveGround__forecast_duration_18:00:00.zarr.json"
ds_t2m_danra_fc = xr.open_zarr(f"reference::{fp_zarr_json}", consolidated=False)
# Select the relevant forecast data and adjust the coordinates
# da_t2m_danra_fc: xr.DataArray = (
#     ds_t2m_danra_fc.sel(
#         analysis_time=ANALYSIS_TIME,
#         time=slice(
#             ANALYSIS_TIME + np.timedelta64(3, "h"),
#             ANALYSIS_TIME + np.timedelta64(18, "h"),
#             3,
#         ),
#         # time=slice(datetime(2022, 1, 1), datetime(2022, 1, 2), timedelta(hours=3)),
#     )
#     .drop_vars("analysis_time")
#     .t
# )
# da_t2m_danra_fc: xr.DataArray = ds_t2m_danra_fc.t
da_t2m_danra_fc = ds_t2m_danra_fc.isel(analysis_time=slice(4)).t

danra_version = "v0.5.0"
# path to copies of processed DANRA datasets on scale.dmi.dk:
fp_root = Path(f"/dmidata/projects/cloudphysics/danra/data/{danra_version}")

ds_danra_analysis_hl = xr.open_zarr(fp_root / "single_levels.zarr")
ds_danra_analysis_hl.attrs["suite_name"] = "danra"
END_TIME = da_t2m_danra_fc["time"].max().values
da_t2m_danra_analysis = ds_danra_analysis_hl["t2m"].sel(
    time=slice(ANALYSIS_TIME + np.timedelta64(3, "h"), END_TIME)
)
# da_t2m_danra_fc = da_t2m_danra_fc.rename(da_t2m_danra_analysis.name)

# fig, ax = plt.subplots(figsize=(8, 10), subplot_kw={"projection": ccrs.AlbersEqualArea(central_longitude=11.0, central_latitude=55.0)})
# da_t2m_danra_analysis = da_t2m_danra_analysis.assign_coords(datasource="forecast")
da_t2m_danra_fc = da_t2m_danra_fc.assign_coords(
    datasource="forecast",
    x=da_t2m_danra_analysis.x,
    y=da_t2m_danra_analysis.y,
    lon=da_t2m_danra_analysis.lon,
    lat=da_t2m_danra_analysis.lat,
)
# __import__("ipdb").set_trace()
# da_t2m_danra_analysis = da_t2m_danra_analysis.drop_attrs(deep=True)
# da_t2m_danra_fc = da_t2m_danra_fc.drop_attrs(deep=True)
# __import__("ipdb").set_trace()
# da_t2m_danra_analysis, da_t2m_danra_fc = xr.align(
#     da_t2m_danra_analysis, da_t2m_danra_fc, join="right"
# )
ax = plot_single_metric_timeseries(
    da_t2m_danra_analysis,
    da_t2m_danra_fc,
    time_axis="UTC",
    hue="analysis_time",
    # time_operation=mean,
    # time_op_kwargs={"dim": "analysis_time"},
)
plt.show()
# ax = plot_single_metric_gridded_map(da_t2m_danra_analysis, da_t2m_danra_fc, time_selection=datetime(2022, 1, 1), axes=ax, xarray_plot_kwargs={"transform": ccrs.PlateCarree()})
# ax.coastlines(linewidth=0.5, color="black")
# ax.gridlines(draw_labels=["top", "left"], color="gray", alpha=0.5, linestyle="--")
# ax.set_extent(bbox, crs=ccrs.PlateCarree())ax = plot_single_metric_timeseries(da_t2m_danra_analysis, da_t2m_danra_fc, time_axis="elapsed")
# ax = plot_single_metric_gridded_map(da_t2m_danra_analysis, da_t2m_danra_fc, time_selection=datetime(2022, 1, 1), axes=ax, xarray_plot_kwargs={"transform": ccrs.PlateCarree()})
# ax.coastlines(linewidth=0.5, color="black")
# ax.gridlines(draw_labels=["top", "left"], color="gray", alpha=0.5, linestyle="--")
# ax.set_extent(bbox, crs=ccrs.PlateCarree())
