"""
Tests for fractopo_network.
"""
from fractopo.analysis.network import Network

import tests
from fractopo_scripts.simulation import fractopo_network


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
            key in fractopo_network.empty_numerical_desc()
            for key in network.numerical_network_description()
        ]
    )
