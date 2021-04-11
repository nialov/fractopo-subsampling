"""
Tests for cli.py.
"""
import traceback
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

import fractopo_subsampling.cli as cli
import fractopo_subsampling.network_scripts as network_scripts
import fractopo_subsampling.schema as schema
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
        assert result.exc_info is not None
        err_type, val, tb = result.exc_info
        print(err_type, val)
        traceback.print_tb(tb)
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
        assert result.exc_info is not None
        err_type, val, tb = result.exc_info
        print(err_type, val)
        traceback.print_tb(tb)
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
        assert result.exc_info is not None
        err_type, val, tb = result.exc_info
        print(err_type, val)
        traceback.print_tb(tb)
        assert False

    args = [results_path_str, save_path_str]
    runner = CliRunner()
    result = runner.invoke(cli.gatherbase, args=args)
    if not result.exit_code == 0:
        assert result.exc_info is not None
        err_type, val, tb = result.exc_info
        print(err_type, val)
        traceback.print_tb(tb)
        assert False


@pytest.mark.parametrize(
    ",".join(
        [
            "traces_path_str",
            "area_path_str",
            "results_path_str",
            "other_results_path_str",
            "coverage_path_str",
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
        ]
        if len(arg) > 0
    ]
    result = runner.invoke(
        cli.subsample,
        args=args,
    )
    if not result.exit_code == 0:
        assert result.exc_info is not None
        err_type, val, tb = result.exc_info
        print(err_type, val)
        traceback.print_tb(tb)
        assert False

    globbed = list(Path(results_path_str).parent.glob("*.csvtest"))
    for result in globbed:
        df = network_scripts.read_csv(result)
        result.unlink()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        schema.describe_df_schema.validate(df)
