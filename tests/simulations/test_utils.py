"""
Tests for fractopo_scripts.simulation.utils.
"""
import numpy as np
import pandas as pd
import pytest

import fractopo_scripts.simulation.utils as utils
import tests


def test_random_sample_of_circles():
    """
    Test random_sample_of_circles.
    """
    df = pd.DataFrame(
        [
            {"key": "a", "value": 1},
            {"key": "a", "value": 1},
            {"key": "b", "value": 2},
            {"key": "b", "value": 2},
            {"key": "b", "value": 2},
            {"key": "b", "value": 2},
            {"key": "b", "value": 2},
            {"key": "b", "value": 2},
            {"key": "b", "value": 1000},
        ]
    ).astype({"key": "category"})
    grouped = df.groupby(by="key")
    circle_names_with_diameter = {"a": 50, "b": 1}
    single_result = utils.random_sample_of_circles(grouped, circle_names_with_diameter)

    assert isinstance(single_result, list)
    assert all([isinstance(val, pd.Series) for val in single_result])

    results = [
        utils.random_sample_of_circles(grouped, circle_names_with_diameter)
        for _ in range(100)
    ]
    name_counts = dict()
    for result in results:
        for srs in result:
            srs_key = srs["key"]
            name_counts[srs_key] = (
                1 if srs_key not in name_counts else name_counts[srs_key] + 1
            )
    assert name_counts["a"] > name_counts["b"]


def test_aggregate_chosen_manual():
    """
    Test aggregate_chosen.
    """
    chosen_dicts = [
        {"area": 1, "intensity": 5},
        {"area": 10, "intensity": 5},
    ]
    chosen = [pd.Series(cd) for cd in chosen_dicts]
    params_with_func = {"intensity": "mean"}
    result = utils.aggregate_chosen(chosen, params_with_func)
    assert isinstance(result, dict)
    assert all([isinstance(val, str) for val in result])
    assert all([isinstance(val, (float, int)) for val in result.values()])
    assert np.isclose(result["intensity"], 5)


@pytest.mark.parametrize(
    "chosen,params_with_func,assume_result", tests.test_aggregate_chosen_params()
)
def test_aggregate_chosen(chosen, params_with_func, assume_result):
    """
    Test aggregate_chosen with pytest params.
    """
    result = utils.aggregate_chosen(chosen, params_with_func)
    assert isinstance(result, dict)
    for key in result:
        if key not in assume_result:
            continue
        assert np.isclose(result[key], assume_result[key])


# Hello
