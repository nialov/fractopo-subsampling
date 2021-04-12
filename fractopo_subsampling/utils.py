"""
General utilities for subsampling.
"""
import csv
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum, unique
from itertools import compress, count
from pathlib import Path
from subprocess import check_call
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas.core.groupby.generic import DataFrameGroupBy


class Utils:

    """
    General selection, column names, etc. data.
    """

    radius = "radius"
    relative_coverage = "relative coverage"
    name = "name"
    trace_power_law_vs_lognormal_r = "trace power_law vs. lognormal R"
    trace_power_law_vs_lognormal_p = "trace power_law vs. lognormal p"

    params_with_func = {
        # "Fracture Intensity (Mauldon)": "mean",
        "Connections per Branch": "mean",
        "trace power_law exponent": "mean",
        "branch power_law exponent": "mean",
        "Fracture Intensity P21": "mean",
        "Number of Traces": "sum",
        "Number of Branches": "sum",
        "radius": "sum",
        "I": "sum",
        "Y": "sum",
        "X": "sum",
        "Trace Boundary 1 Intersect Count": "sum",
        "Trace Boundary 2 Intersect Count": "sum",
        "Trace Boundary 0 Intersect Count": "sum",
    }

    renamed_params = {
        "trace power_law exponent": "Trace Power-law Exponent",
        "branch power_law exponent": "Branch Power-law Exponent",
    }

    selected_params = {
        "Connections per Branch",
        "trace power_law exponent",
        "branch power_law exponent",
        "Fracture Intensity P21",
    }


@unique
class Skip(Enum):

    """
    Enums for skip options.
    """

    valid = "valid"
    invalid = "invalid"
    empty = "empty"


SCRIPTS_RUN_CMD = "python3 -m fractopo_subsampling"


def param_renamer(param: str):
    """
    Rename param for nicer plotting name.

    If no rename in renamed_params is defined no renaming is done.
    """
    try:
        return Utils.renamed_params[param]
    except KeyError:
        return param


def random_sample_of_circles(
    grouped: DataFrameGroupBy,
    circle_names_with_diameter: Dict[str, int],
    min_circles: int = 1,
    max_circles: int = None,
    min_area: float = 0.0,
    max_area: float = None,
) -> List[pd.Series]:
    """
    Get a random sample of circles from grouped subsampled data.

    Both the amount of overall circles and which circles within each group
    is random. Data is grouped by target area name.
    """
    if max_circles is not None:
        assert max_circles >= min_circles

    # Area names
    names = list(grouped.groups.keys())

    # Get area of the base circles corresponding to area name
    areas = [np.pi * (circle_names_with_diameter[name] / 2) ** 2 for name in names]

    # All indexes
    idxs = list(range(0, len(grouped)))

    # "Randomly" choose how many circles
    # Is constrained by given min_circles and max_circles
    how_many = random.randint(
        min_circles, len(grouped) if max_circles is None else max_circles
    )

    # Collect indexes of base circles
    which_idxs = []
    for _ in range(how_many):
        compressor = [idx not in which_idxs for idx in idxs]
        possible_idxs = list(compress(idxs, compressor))
        possible_areas = list(compress(areas, compressor))
        choice = random.choices(population=possible_idxs, weights=possible_areas, k=1)[
            0
        ]
        which_idxs.append(choice)

    assert len(which_idxs) == how_many

    # Collect the Series that are chosen
    chosen: List[pd.Series] = []

    # Iterate over the DataFrameGroupBy dataframe groups
    for idx, (_, group) in enumerate(grouped):

        # Skip if not chosen base circle previously
        if idx not in which_idxs:
            continue
        # radii = group["radius"].values
        # radii_weights = normalize_and_invert_weights(
        #     radii,
        #     max_value=(Utils.circle_names_with_diameter[str(name)] / 2)
        #     if name in Utils.circle_names_with_diameter
        #     else None,
        # )
        # radii_weights = radii

        # Make continous index from 0
        indexer = count(0)
        indexes = [next(indexer) for _ in group.index.values]

        # If min_area or max_area are given, the choices are filtered
        # accordingly
        if min_area > 0 or max_area is not None:

            # Get circle areas
            areas = group["area"].to_numpy()

            # Solve max_area
            max_area = np.max(areas) if max_area is None else max_area

            # Filter out areas that do not fit within the range
            area_compressor = [min_area <= area <= max_area for area in areas]

            # Filter out indexes accordingly
            indexes = list(compress(indexes, area_compressor))

        assert len(indexes) > 0
        # Choose from indexes
        choice = random.choices(population=indexes, k=1)[0]

        # Get the Series at choice index
        srs = group.iloc[choice]
        assert isinstance(srs, pd.Series)

        # Collect
        chosen.append(srs)

    assert len(chosen) == how_many

    # Return chosen subsampled circles from base circles
    return chosen


def normalize_and_invert_weights(
    weights: Sequence[float], max_value: Optional[float] = None
) -> Sequence[float]:
    """
    Normalize a list of weights and invert them.
    """
    # Return if empty
    if len(weights) == 0:
        return weights

    # Get actual max value of weights if none is given
    if max_value is None:
        max_value = max(weights)
    else:
        assert max_value >= max(weights)

    # Normalize and invert
    return [1 - (val / max_value) for val in weights]


def numpy_to_python_type(value):
    """
    Convert to Python type from numpy with .item().
    """
    try:
        return value.item()
    except AttributeError:
        return value


def aggregate_chosen(
    chosen: List[pd.Series], params_with_func: Dict[str, str]
) -> Dict[str, Any]:
    """
    Aggregate a collection of subsampled circles for params.

    Weights averages by the area of each subsampled circle.
    """
    total_area = numpy_to_python_type(sum([srs["area"] for srs in chosen]))
    assert isinstance(total_area, (float, int))
    total_count = len(chosen)
    values = dict(area=total_area, circle_count=total_count)
    for param, func in params_with_func.items():
        if func == "mean":
            func_value = np.average(
                [srs[param] for srs in chosen], weights=[srs["area"] for srs in chosen]
            )
        elif func == "sum":
            func_value = np.array([srs[param] for srs in chosen]).sum()
        else:
            raise ValueError("Expected mean or sum.")
        func_value = numpy_to_python_type(func_value)
        assert isinstance(func_value, (float, int))
        values[param] = func_value
    return values


def constrain_radius(
    names: np.ndarray, radiuses: np.ndarray, circle_names_with_diameter: Dict[str, int]
) -> List[bool]:
    """
    Constrain dataset radiuses to one fourth of the full diameter.
    """
    constrained = []
    for name, radius in zip(names, radiuses):
        assert name in circle_names_with_diameter
        full_radius = circle_names_with_diameter[name]
        constrained.append(full_radius / 4 >= radius)
    assert len(constrained) == len(names) == len(radiuses)
    return constrained


def radius_to_area(radius: float):
    """
    Convert circle radius to area.
    """
    return np.pi * radius ** 2


def area_to_radius(area: float):
    """
    Convert circle area to radius.
    """
    return np.sqrt(area / np.pi)


def filter_dataframe(
    df: Union[gpd.GeoDataFrame, pd.DataFrame],
    filter_names: List[str],
    filter_radius: Tuple[float, float],
    relative_coverage_threshold: float,
) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Filter (Geo)DataFrame to input names and thresholds.
    """
    start_length = df.shape[0]
    assert df.shape[0] > 0
    named_df = df.loc[np.isin(df[Utils.name], filter_names)]
    print(f"Name filtered: {start_length - named_df.shape[0]}")
    assert named_df.shape[0] > 0
    radius_df = named_df.loc[
        [filter_radius[0] <= val <= filter_radius[1] for val in named_df[Utils.radius]]
    ]
    print(f"Radius filtered: {named_df.shape[0] - radius_df.shape[0] }")
    assert radius_df.shape[0] > 0
    coverage_df = radius_df.loc[
        radius_df[Utils.relative_coverage] < relative_coverage_threshold
    ]
    print(f"Coverage filtered: {radius_df.shape[0] - coverage_df.shape[0] }")
    assert coverage_df.shape[0] > 0
    return coverage_df


def paper_figsize(
    multiplier: float,
    paper_height=11.7,
    paper_width=8.27,
) -> Tuple[float, float]:
    """
    Get figsize for A4.
    """
    return (paper_width, min([paper_height, paper_height * multiplier]))


def label_point(
    xs: Sequence[float], ys: Sequence[float], vals: Sequence, ax: Axes, **text_kwargs
):
    """
    Label points in plot.
    """
    [ax.text(x + 0.02, y, str(val), **text_kwargs) for x, y, val in zip(xs, ys, vals)]


def cached_subsampling(
    dataframe_grouped: DataFrameGroupBy,
    iterations: int,
    savepath: Path,
    circle_names_with_diameter: Dict[str, int],
    subsample_area_limits: Optional[Tuple[float, float]] = None,
):
    """
    Perform subsampling.
    """
    if savepath.exists():
        agg_df = pd.read_csv(savepath)
    else:
        agg_df = pd.DataFrame(
            [
                aggregate_chosen(
                    random_sample_of_circles(
                        dataframe_grouped,
                        circle_names_with_diameter,
                        min_area=subsample_area_limits[0],
                        max_area=subsample_area_limits[1],
                    ),
                    params_with_func=Utils.params_with_func,
                )
                for _ in range(iterations)
            ]
        )
        agg_df.to_csv(savepath, index=False)
    return agg_df


def collect_paths(
    csv_path: str,
    skip: List[Literal[Skip.empty, Skip.valid, Skip.invalid]],
    filter: str = "",
) -> Tuple[List[Path], List[Path], List[Tuple[str, str, float]]]:
    """
    Collect trace and area paths from relations.csv file.
    """
    if not all([val in Skip for val in skip]):
        raise ValueError(f"Expected skip vals to be one of {Skip}.")
    traces_paths, area_paths, marks = [], [], []
    with Path(csv_path).open("r") as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            base_path = "{}/{}/{}/{}"
            area_path = Path(
                base_path.format(row[2], "areas", row[3], row[0] + ".gpkg")
            ).resolve()
            traces_path = Path(
                base_path.format(row[2], "traces", row[3], row[1] + ".gpkg")
            ).resolve()
            if not (area_path.exists() and traces_path.exists()):
                continue
            if row[4] == "True" and Skip.valid in skip:
                # Do not collect valid.
                continue
            if row[4] == "False" and Skip.invalid in skip:
                # Do not collect invalid.
                continue
            if row[5] == "True" and Skip.empty in skip:
                # Do not collect empty.
                continue
            if len(filter) > 0 and filter not in area_path.stem:
                # Only collect area paths that fit filter.
                continue

            traces_paths.append(traces_path)
            area_paths.append(area_path)
            # valid, empty, radius
            diameter = float(row[-1])
            marks.append((row[4], row[5], diameter))
    return traces_paths, area_paths, marks


def async_run(all_args: List[Sequence[str]]):
    """
    Run command line tasks in parallel.
    """
    # If max_workers is None, the amount will be equal to the amount of
    # processors.
    with ProcessPoolExecutor(max_workers=None) as executor:

        # Submit tasks, collect futures
        futures = [executor.submit(check_call, args) for args in all_args]

        results = []

        # Collect results as they complete
        for result in as_completed(futures):

            try:
                results.append(result.result())
            except Exception as err:

                # Log exceptions to stder (shouldnt occur)
                logging.error(err)
