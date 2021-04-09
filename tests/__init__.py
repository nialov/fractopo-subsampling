"""
Tests for fractopo_subsampling.

Contains most test parameters.
"""
import tempfile
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from fractopo.general import read_geofile
from shapely.geometry import Point

flato_traces_path_str = "tests/sample_data/Flato_20m_1_traces.gpkg"
flato_area_path_str = "tests/sample_data/Flato_20m_1_area.gpkg"
flato_traces_path = Path(flato_traces_path_str)
flato_area_path = Path(flato_area_path_str)
shoreline_path = Path("misc/shoreline.geojson")


@lru_cache(maxsize=None)
def flato_traces_gdf(flato_traces_path: Path = flato_traces_path) -> gpd.GeoDataFrame:
    """
    Get flato traces GeoDataFrame.
    """
    traces = read_geofile(flato_traces_path)
    return traces


@lru_cache(maxsize=None)
def flato_area_gdf(flato_area_path: Path = flato_area_path) -> gpd.GeoDataFrame:
    """
    Get flato area GeoDataFrame.
    """
    area = read_geofile(flato_area_path)
    return area


@lru_cache(maxsize=None)
def shoreline_gdf(shoreline_path: Path = shoreline_path) -> gpd.GeoDataFrame:
    """
    Get shoreline GeoDataFrame.
    """
    shoreline = read_geofile(shoreline_path)
    return shoreline


def test_main_params():
    """
    Parameters for cli.main tests.
    """
    return [
        ([]),
        (["--help"]),
    ]


def test_baseanalyze_params():
    """
    Parameters for cli.baseanalyze tests.
    """
    return [
        (["--help"]),
        (
            [
                flato_traces_path_str,
                flato_area_path_str,
                f"{tempfile.mkdtemp()}",
                f"{tempfile.mkdtemp()}",
                flato_area_path_str,
                "25.0",
            ]
        ),
    ]


def test_sim_params():
    """
    Parameters for cli.sim tests.
    """
    return [
        (
            flato_traces_path_str,
            flato_area_path_str,
            f"{tempfile.mktemp()}",
            f"{tempfile.mkdtemp()}",
            flato_area_path_str,
            "--how-many",
            "1",
            "",
        ),
        (
            flato_traces_path_str,
            flato_area_path_str,
            f"{tempfile.mktemp()}",
            f"{tempfile.mkdtemp()}",
            flato_area_path_str,
            "--how-many",
            "2",
            "",
        ),
        (
            flato_traces_path_str,
            flato_area_path_str,
            f"{tempfile.mktemp(suffix='.csvtest')}",
            f"{tempfile.mkdtemp()}",
            flato_area_path_str,
            "--how-many",
            "1",
            "--hashname",
        ),
    ]


def test_baseanalyze_with_gather_params():
    """
    Parameters for baseanalyze_with_gather test.
    """
    params = []
    for (
        traces_path_str,
        area_path_str,
        results_path_str,
        other_results_path_str,
        overwrite,
        save_path_str,
        circle_radius,
    ) in zip(
        [flato_traces_path_str],
        [flato_area_path_str],
        [tempfile.mkdtemp()],
        [tempfile.mkdtemp()],
        [""],
        [f"{tempfile.mktemp(suffix='.csv')}"],
        ["25.0"],
    ):
        param = (
            traces_path_str,
            area_path_str,
            results_path_str,
            other_results_path_str,
            area_path_str,
            circle_radius,
            overwrite,
            save_path_str,
        )
        assert len(param) == 8
        params.append(param)
    return params


@lru_cache(maxsize=None)
def test_aggregate_chosen_params():
    """
    Params for test_aggregate_chosen_manual.
    """

    def make_param(chosen_dicts: list, params_with_func: dict, assume_result: dict):
        return ([pd.Series(cd) for cd in chosen_dicts], params_with_func, assume_result)

    return [
        make_param(
            [
                {"area": 1, "intensity": 5},
                {"area": 10, "intensity": 5},
            ],
            {"intensity": "mean"},
            assume_result={"intensity": 5},
        ),
        make_param(
            [
                {"area": 1, "intensity": 5, "hello": 1},
                {"area": 10, "intensity": 5, "hello": 10},
            ],
            {"intensity": "mean", "hello": "sum"},
            assume_result={"intensity": 5, "hello": 11},
        ),
        make_param(
            [
                {"area": 1, "intensity": 0, "hello": 1},
                {"area": 2, "intensity": 1, "hello": 10},
            ],
            {"intensity": "mean", "hello": "sum"},
            assume_result={"intensity": 0.66666666, "hello": 11},
        ),
    ]


@lru_cache(maxsize=None)
def test_base_circle_id_coords_params():
    """
    Params for test_base_circle_id_coords.
    """
    dfs = [
        pd.DataFrame(
            {
                "x": [10, 15, 123, -4],
                "y": [2, 1, 1, -4],
                "name": ["a", "b", "c", "d"],
                "radius": [25.0, 25.0, 25.0, 25.0],
                "relative coverage": [0.0, 0.0, 0.0, 0.001],
            }
        )
    ]

    processed = []
    for df in dfs:
        df["area"] = np.pi * df["radius"] ** 2
        processed.append(df)
    return processed


@lru_cache(maxsize=None)
def point_analysis_gdf():
    """
    Params for test_base_circle_id_coords.
    """
    gdfs = []
    for df in test_base_circle_id_coords_params():
        gdf = gpd.GeoDataFrame(df)
        gdf["points"] = [Point(x, y) for x, y in zip(df["x"].values, df["y"].values)]
        gdf = gdf.set_geometry(col="points", drop=True)
        gdfs.append(gdf)
    return gdfs


@lru_cache(maxsize=None)
def test_label_ids_to_map_params():
    """
    Params for test_label_ids_to_map.
    """
    return [[(-4, -4, "d"), (10, 2, "a"), (15, 1, "b"), (123, 1, "c")]]


@lru_cache(maxsize=None)
def test_preprocess_analysis_points_params():
    """
    Params for preprocess_analysis_points.
    """
    filter_radius = 5.0, 51.0
    relative_coverage_threshold = 0.11
    params = []
    for gdf in point_analysis_gdf():
        circle_names_with_diameter = {str(key): 50.0 for key in gdf["name"].values}
        params.append(
            (
                gdf,
                circle_names_with_diameter,
                filter_radius,
                relative_coverage_threshold,
            )
        )

    return params
