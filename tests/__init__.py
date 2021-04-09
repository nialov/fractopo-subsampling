"""
Tests for fractopo_subsampling.

Contains most test parameters.
"""
import tempfile
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import pandas as pd
from fractopo.general import read_geofile

flato_traces_path_str = "tests/sample_data/Flato_20m_1_traces.gpkg"
flato_area_path_str = "tests/sample_data/Flato_20m_1_area.gpkg"
flato_traces_path = Path(flato_traces_path_str)
flato_area_path = Path(flato_area_path_str)


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
