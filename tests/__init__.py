"""
Tests for fractopo_scripts.

Contains most test parameters.
"""
import tempfile


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
                "tests/sample_data/Flato_20m_1_traces.gpkg",
                "tests/sample_data/Flato_20m_1_area.gpkg",
                f"{tempfile.mkdtemp()}",
                f"{tempfile.mkdtemp()}",
            ]
        ),
    ]


def test_sim_params():
    """
    Parameters for cli.sim tests.
    """
    return [
        (["--help"]),
        (
            [
                "tests/sample_data/Flato_20m_1_traces.gpkg",
                "tests/sample_data/Flato_20m_1_area.gpkg",
                f"{tempfile.mktemp()}",
                f"{tempfile.mkdtemp()}",
                "tests/sample_data/coverage_in_target_areas.gpkg",
            ]
        ),
        (
            [
                "tests/sample_data/Flato_20m_1_traces.gpkg",
                "tests/sample_data/Flato_20m_1_area.gpkg",
                f"{tempfile.mktemp()}",
                f"{tempfile.mkdtemp()}",
                "tests/sample_data/coverage_in_target_areas.gpkg",
                "--how-many",
                "2",
            ]
        ),
    ]
