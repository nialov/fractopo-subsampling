"""
Tests for network_scripts.
"""
from fractopo.analysis.network import Network

import tests
from fractopo_subsampling import network_scripts


def test_empty_numerical_desc_manual():
    """
    Test that all keys are valid.
    """
    network = Network(
        trace_gdf=tests.flato_traces_gdf(),
        area_gdf=tests.flato_area_gdf(),
        snap_threshold=0.001,
        determine_branches_nodes=True,
        truncate_traces=True,
    )
    assert all(
        [
            key in network_scripts.empty_numerical_desc()
            for key in network.numerical_network_description()
        ]
    )


def test_plaifiny_rose_plot_manual():
    """
    Test plainify_rose_plot.
    """
    network = Network(
        trace_gdf=tests.flato_traces_gdf(),
        area_gdf=tests.flato_area_gdf(),
        snap_threshold=0.001,
        determine_branches_nodes=False,
        truncate_traces=True,
    )

    _, fig, ax = network.plot_trace_azimuth()
    network_scripts.plainify_rose_plot(fig, ax)
