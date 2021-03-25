"""
General utilities for simulation.
"""
import random
from itertools import compress, count
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas.core.groupby.generic import DataFrameGroupBy


class Utils:

    """
    General selection, column names, etc. data.
    """

    circle_names_with_diameter = {
        "Getaberget_20m_4_3_area": 50,
        "Getaberget_20m_9_2_area": 50,
        "Getaberget_20m_8_3_area": 50,
        "Getaberget_20m_7_1_area": 50,
        "Getaberget_20m_7_2_area": 20,  # 20 m
        "Getaberget_20m_5_1_area": 50,
        "Getaberget_20m_2_1_area": 40,  # 40 m
        "Getaberget_20m_2_2_area": 50,
        "Getaberget_20m_1_1_area": 50,
        "Getaberget_20m_1_2_area": 40,  # 40 m
        "Getaberget_20m_1_3_area": 20,  # 10 m
        "Getaberget_20m_1_4_area": 50,
        "Havsvidden_20m_1_area": 50,
    }

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
    }

    base_circle_ids_csv_path = Path("../results/base_circle_ids.csv")
    base_circle_reference_value_csv_path = Path("../results/base_reference_values.csv")

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
    Get a random sample of circles from grouped simulation data.

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
            area_compressor = [min_area > area > max_area for area in areas]

            # Filter out indexes accordingly
            indexes = list(compress(indexes, area_compressor))

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
    chosen: List[pd.Series], params_with_func: Dict[str, Literal["mean", "sum"]]
) -> Dict[str, Union[float, int]]:
    """
    Aggregate a collection of simulation circles for params.

    Weights averages by the area of each simulation circle.
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
