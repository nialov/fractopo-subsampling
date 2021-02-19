"""
Tests for simulation.cli.py.
"""
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

import fractopo_scripts.simulation.cli as cli
import fractopo_scripts.simulation.fractopo_network as fn
import fractopo_scripts.simulation.schema as schema
import tests


@pytest.mark.parametrize("params", tests.test_main_params())
def test_main(params: list):
    """
    Test main entrypoint.
    """
    assert isinstance(params, list)
    runner = CliRunner()
    result = runner.invoke(cli.main, args=params)
    if not result.exit_code == 0:
        print(result.stdout)
        assert False


@pytest.mark.parametrize("params", tests.test_baseanalyze_params())
def test_baseanalyze(params: list):
    """
    Test baseanalyze entrypoint.
    """
    assert isinstance(params, list)
    runner = CliRunner()
    result = runner.invoke(cli.baseanalyze, args=params)
    if not result.exit_code == 0:
        print(result.stdout)
        assert False


@pytest.mark.parametrize(
    "traces_path_str,area_path_str,results_path_str,"
    "other_results_path_str,coverage_path_str,"
    "circle_radius,overwrite,save_path_str",
    tests.test_baseanalyze_with_gather_params(),
)
def test_baseanalyze_with_gather(
    traces_path_str,
    area_path_str,
    results_path_str,
    other_results_path_str,
    coverage_path_str,
    circle_radius,
    overwrite,
    save_path_str,
):
    """
    Test baseanalyze and gatherbase entrypoints.
    """
    args = [
        arg
        for arg in [
            traces_path_str,
            area_path_str,
            results_path_str,
            other_results_path_str,
            coverage_path_str,
            circle_radius,
            overwrite,
        ]
        if len(arg) > 0
    ]
    runner = CliRunner()
    result = runner.invoke(cli.baseanalyze, args=args)
    if not result.exit_code == 0:
        print(result.stdout)
        assert False

    args = [results_path_str, save_path_str]
    runner = CliRunner()
    result = runner.invoke(cli.gatherbase, args=args)
    if not result.exit_code == 0:
        print(result.stdout)
        assert False


@pytest.mark.parametrize(
    ",".join(
        [
            "traces_path_str",
            "area_path_str",
            "results_path_str",
            "other_results_path_str",
            "coverage_path_str",
            "how_many",
            "how_many_count",
            "hashname",
        ]
    ),
    tests.test_sim_params(),
)
def test_sim(
    traces_path_str,
    area_path_str,
    results_path_str,
    other_results_path_str,
    coverage_path_str,
    how_many,
    how_many_count,
    hashname,
):
    """
    Test sim click entrypoint.
    """
    runner = CliRunner()
    args = [
        arg
        for arg in [
            traces_path_str,
            area_path_str,
            results_path_str,
            other_results_path_str,
            coverage_path_str,
            how_many,
            how_many_count,
            hashname,
        ]
        if len(arg) > 0
    ]
    result = runner.invoke(
        cli.sim,
        args=args,
    )
    if not result.exit_code == 0:
        print(result.stdout)
        assert False

    if len(hashname) == 0:
        assert Path(results_path_str).exists()
        df = fn.read_csv(Path(results_path_str))
        assert isinstance(df, pd.DataFrame)
        schema.describe_df_schema.validate(df)
    else:
        globbed = list(Path(results_path_str).parent.glob("*.csvtest"))
        if not len(globbed) == 1:
            for path in globbed:
                path.unlink()
            assert False
        df = fn.read_csv(globbed[0])
        globbed[0].unlink()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        schema.describe_df_schema.validate(df)
