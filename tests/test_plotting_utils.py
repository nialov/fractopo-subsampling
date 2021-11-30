"""
Tests for plotting_utils.py.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pytest
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

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


@pytest.mark.parametrize("group", tests.test_plot_group_pair_boxplots_params())
def test_plot_group_pair_boxplots(make_plot, group):
    """
    Test plot_group_pair_boxplots.
    """
    group_col_second = tests.group_col_second
    param = tests.FRACTURE_INTENSITY_P21
    group_second_labels = ("1-2", "2-3")
    multip_diff = 1.1
    outlier_proportion_threshold = 1.0
    reference_value_dict = {param: 2.5}
    i = 0
    j = 0
    _, ax = make_plot
    ax_gen = (an_ax for an_ax in [ax])
    cc_gen = (val for val in ["2-4"])
    result = plotting_utils.plot_group_pair_boxplots(
        group=group,
        group_col_second=group_col_second,
        group_second_labels=group_second_labels,
        reference_value_dict=reference_value_dict,
        multip_diff=multip_diff,
        outlier_proportion_threshold=outlier_proportion_threshold,
        i=i,
        j=j,
        ax_gen=ax_gen,
        cc_gen=cc_gen,
        param=param,
    )

    assert isinstance(result, Axes)


@pytest.mark.parametrize("aggregate_df", tests.test_plot_group_pair_boxplots_params())
def test_grouped_boxplots(aggregate_df):
    """
    Test plot_group_pair_boxplots.
    """
    group_col_first = tests.group_col_first
    group_col_second = tests.group_col_second
    group_first_labels = ("1-2", "2-3")
    group_second_labels = ("1-2", "2-3")
    multip_diff = 1.1
    outlier_proportion_threshold = 1.0
    reference_value_dict = {tests.param: 2.5}
    try:
        result = plotting_utils.grouped_boxplots(
            aggregate_df=aggregate_df,
            reference_value_dict=reference_value_dict,
            group_col_first=group_col_first,
            group_col_second=group_col_second,
            group_first_labels=group_first_labels,
            group_second_labels=group_second_labels,
            multip_diff=multip_diff,
            outlier_proportion_threshold=outlier_proportion_threshold,
        )
    except Exception:
        plt.close()
        raise
    plt.close()

    assert isinstance(result, Figure)


@pytest.mark.parametrize("agg_df", tests.test_plot_group_pair_boxplots_params())
def test_group_pair_counts(agg_df):
    """
    Test group_pair_counts.
    """
    result = plotting_utils.plot_group_pair_counts(
        agg_df=agg_df,
        x=tests.group_col_first,
        hue=tests.group_col_second,
        xlabel="xlabel",
        ylabel="ylabel",
        title="sometitle",
        legend_title="legend_title",
    )

    assert isinstance(result, Figure)
