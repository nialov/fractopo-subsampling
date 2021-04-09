"""
Tests for plotting_utils.py.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes._axes import Axes

import fractopo_subsampling.plotting_utils as plotting_utils
import tests


@pytest.fixture
def make_plot():
    """
    Make dummy plot.
    """
    fig, ax = plt.subplots()
    yield fig, ax

    plt.close()


@pytest.mark.parametrize("filtered", tests.test_base_circle_id_coords_params())
def test_base_circle_id_coords(filtered):
    """
    Test base_circle_id_coords.
    """
    result = plotting_utils.base_circle_id_coords(filtered)

    assert isinstance(result, list)


@pytest.mark.parametrize("coords", tests.test_label_ids_to_map_params())
def test_label_ids_to_map(coords, make_plot):
    """
    Test label_ids_to_map.
    """
    _, ax = make_plot
    result = plotting_utils.label_ids_to_map(coords, ax)

    assert isinstance(result, Axes)


@pytest.mark.parametrize("gdf", tests.point_analysis_gdf())
def test_plot_base_circle_map(gdf):
    """
    Test plot_base_circle_map.
    """
    shoreline = tests.shoreline_gdf()
    ax = plotting_utils.plot_base_circle_map(filtered=gdf, shoreline=shoreline)

    assert isinstance(ax, Axes)


@pytest.mark.parametrize(
    "analysis_points,circle_names_with_diameter,"
    "filter_radius,relative_coverage_threshold",
    tests.test_preprocess_analysis_points_params(),
)
def test_preprocess_analysis_points(
    analysis_points,
    circle_names_with_diameter,
    filter_radius,
    relative_coverage_threshold,
):
    """
    Test preprocess_analysis_points.
    """
    result = plotting_utils.preprocess_analysis_points(
        analysis_points,
        circle_names_with_diameter,
        filter_radius=filter_radius,
        relative_coverage_threshold=relative_coverage_threshold,
    )

    assert isinstance(result, gpd.GeoDataFrame)


def test_colorgen():
    """
    Test color generator.
    """
    my_colors = "red", "black", "green"

    for idx, color in enumerate(plotting_utils.colorgen(my_colors)):
        assert isinstance(color, str)
        assert color in my_colors

        if idx > 15:
            break
