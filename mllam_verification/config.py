from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from pydantic.dataclasses import dataclass as pydantic_dataclass


@pydantic_dataclass
class Config:
    schema_version: str
    inputs: "Inputs"
    methods: List[str]
    output: "Output"


@pydantic_dataclass
class Dataset:
    path: Path


@pydantic_dataclass
class TimeRange:
    start: datetime
    end: datetime
    step: timedelta


@pydantic_dataclass
class Inputs:
    datasets: "Datasets"
    variables: List[str]
    coord_ranges: "CoordRanges"


@pydantic_dataclass
class Datasets:
    initial: Dataset
    target: Dataset
    prediction: Dataset


@pydantic_dataclass
class CoordRanges:
    time: TimeRange


@pydantic_dataclass
class Output:
    path: Path
