"""
Tests for simulation.cli.py.
"""
import pytest
from click.testing import CliRunner

import fractopo_scripts.simulation.cli as cli
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


@pytest.mark.parametrize("params", tests.test_sim_params())
def test_sim(params: list):
    """
    Test sim click entrypoint.
    """
    assert isinstance(params, list)
    runner = CliRunner()
    result = runner.invoke(cli.sim, args=params)
    if not result.exit_code == 0:
        print(result.stdout)
        assert False
