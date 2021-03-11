"""
Fractopo Network sampling scripts.
"""
import logging
from pathlib import Path
from shutil import move
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from fractopo.analysis.network import Network
from fractopo.general import pygeos_spatial_index, safe_buffer
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from shapely.geometry import Point
from shapely.wkt import loads

from fractopo_scripts.simulation.schema import describe_df_schema
from pandera import DataFrameSchema

GEOM_COL = "geometry"


def read_csv(path: Path) -> pd.DataFrame:
    """
    Read csv file with ; separator.
    """
    df = pd.read_csv(path, sep=";", index_col=[0])
    assert isinstance(df, pd.DataFrame)
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """
    Save csv file with ; separator.
    """
    df.to_csv(path, sep=";")


def save_results(gdf: gpd.GeoDataFrame, save_path: Path):
    """
    Save GeoDataFrame results but keep backup if file at path exists.
    """
    if save_path.exists():
        logging.debug("Found existing. Creating backup.")
        move(save_path, save_path.parent / f"{save_path.name}.backup")
    gdf.to_file(save_path, driver="GPKG")


def gather_results(results_path: Path) -> gpd.GeoDataFrame:
    """
    Gather base analysis results.
    """
    if not (results_path.exists() and results_path.is_dir()):
        raise NotADirectoryError(f"Expected {results_path} dir to exist.")
    df = pd.DataFrame(
        [pd.read_pickle(result_path) for result_path in results_path.glob("*.pickle")]
    )
    df[GEOM_COL] = [loads(wkt) for wkt in df[GEOM_COL].values]
    gdf = gpd.GeoDataFrame(df).set_crs(3067)
    return gdf


def gather_sim_results(results_path: Path) -> pd.DataFrame:
    """
    Gather simulation results.
    """
    if not (results_path.exists() and results_path.is_dir()):
        raise NotADirectoryError(f"Expected {results_path} dir to exist.")

    dfs = []
    for path in results_path.glob("*.csv"):
        df = read_csv(path)
        dfs.append(df)

    assert all([isinstance(val, pd.DataFrame) for val in dfs])
    concatted = pd.concat(dfs, ignore_index=True)
    validated = describe_df_schema.validate(concatted)
    return validated


def save_azimuth_bin_data(network: Network, other_results_path: Path, loc_hash: str):
    """
    Save azimuth bin data from a Network.
    """
    trace_bins, _, _ = network.plot_trace_azimuth()
    branch_bins, _, _ = network.plot_branch_azimuth()
    trace_bin_df, branch_bin_df = bins_to_dataframes(trace_bins, branch_bins)
    save_csv(trace_bin_df, other_results_path / f"trace_{loc_hash}.csv")
    save_csv(branch_bin_df, other_results_path / f"branch_{loc_hash}.csv")


def assess_coverage(
    target_centroid: Point, radius: float, coverage_gdf: gpd.GeoDataFrame
) -> float:
    """
    Determine the coverage within a circle.

    Based on manually digitized gpkg of coverage.
    """
    circle = safe_buffer(target_centroid, radius=radius)

    index_intersection = pygeos_spatial_index(coverage_gdf).intersection(circle.bounds)
    candidate_idxs = list(index_intersection if index_intersection is not None else [])
    if len(candidate_idxs) == 0:
        return 0.0
    candidates = coverage_gdf.iloc[candidate_idxs]
    coverage_area = gpd.clip(candidates, circle).area.sum()
    assert isinstance(coverage_area, float)
    return coverage_area


def bins_to_dataframes(trace_bins, branch_bins):
    """
    Transform azimuth bin data to pandas DataFrames.
    """
    trace_bin_df = pd.DataFrame(trace_bins)
    branch_bin_df = pd.DataFrame(branch_bins)
    return trace_bin_df, branch_bin_df


def create_describe_df(
    target_centroid: Point,
    numerical_network_description: dict,
    name: str,
    radius: float,
    amount_of_coverage: float,
    describe_df_schema: DataFrameSchema,
) -> pd.DataFrame:
    """
    Create DataFrame for network.
    """
    describe_df = pd.DataFrame(
        [
            {
                GEOM_COL: target_centroid.wkt,
                **numerical_network_description,
                "name": name,
                "radius": radius,
                "area": np.pi * radius ** 2,
                "coverage": amount_of_coverage,
                "relative coverage": amount_of_coverage / (np.pi * radius ** 2),
            }
        ]
    )
    describe_df = describe_df_schema.validate(describe_df)
    return describe_df


def save_numerical_data(
    numerical_network_description,
    target_centroid,
    radius,
    name,
    results_path: Path,
    amount_of_coverage: float,
):
    """
    Save numerical data of Network sampling.
    """
    curr_df = create_describe_df(
        target_centroid=target_centroid,
        name=name,
        radius=radius,
        numerical_network_description=numerical_network_description,
        amount_of_coverage=amount_of_coverage,
        describe_df_schema=describe_df_schema,
    )

    if results_path.exists():
        df = read_csv(results_path)
        assert isinstance(df, pd.DataFrame)
        concatted = pd.concat([df, curr_df])
    else:
        concatted = curr_df

    save_csv(concatted, results_path)
    concatted.to_pickle(results_path.with_suffix(".pickle"))


def analyze(
    traces: gpd.GeoDataFrame,
    area: gpd.GeoDataFrame,
    name: str,
    other_results_path: Path,
    coverage_gdf: gpd.GeoDataFrame,
    circle_radius: float,
):
    """
    Analyze traces and area for wanted results.

    We want:

    -  Point with all results that are possible
    -  All points in single layer
    """
    # Analyze
    network, description_srs = network_analyze(
        traces, area, name=name, coverage_gdf=coverage_gdf, circle_radius=circle_radius
    )
    plot_and_save_azimuths(network, other_results_path, name)
    save_branches_and_nodes(network, other_results_path)

    assert isinstance(description_srs, pd.Series)
    return description_srs


def save_branches_and_nodes(network: Network, other_results_path: Path):
    """
    Save branches and nodes from Network to other_results_path.
    """
    try:
        network.branch_gdf.to_file(
            other_results_path / f"{network.name}_branches.gpkg", driver="GPKG"
        )
        network.node_gdf.to_file(
            other_results_path / f"{network.name}_nodes.gpkg", driver="GPKG"
        )
    except:
        logging.error(f"Failed to save branches and nodes to {other_results_path}.")


def plainify_rose_plot(fig: Figure, ax: PolarAxes):
    """
    Make plain version of rose plots with only bars.
    """
    ax.set_title("")
    for txt in ax.texts:
        txt.set_visible(False)
    ax.set_rgrids([]), ax.set_thetagrids([]), ax.set_axis_off()
    fig.patch.set_alpha(1)
    ax.patch.set_alpha(1)


def plot_and_save_azimuths(network: Network, other_results_path: Path, name: str):
    """
    Plot azimuth rose plots and save them.
    """
    _, fig, ax = network.plot_trace_azimuth()
    # save normal version
    fig.savefig(other_results_path / f"{name}_trace_azimuths.svg", bbox_inches="tight")
    # Make plain version
    plainify_rose_plot(fig, ax)
    # Save plain version
    fig.savefig(
        other_results_path / f"{name}_trace_azimuths_plain.svg",
        bbox_inches="tight",
        transparent=True,
    )
    # Branch version
    _, fig, _ = network.plot_branch_azimuth()
    fig.savefig(other_results_path / f"{name}_branch_azimuths.svg", bbox_inches="tight")


def network_analyze(
    traces: gpd.GeoDataFrame,
    area: gpd.GeoDataFrame,
    name: str,
    coverage_gdf: gpd.GeoDataFrame,
    circle_radius: float,
) -> Tuple[Network, pd.Series]:
    """
    Analyze traces and area for results.
    """
    network = Network(
        trace_gdf=traces,
        area_gdf=area,
        name=name,
        determine_branches_nodes=True,
        snap_threshold=0.001,
    )
    points = network.representative_points()
    if not len(points) == 1:
        raise ValueError("Expected one target area representative point.")
    point = points[0]
    amount_of_coverage = assess_coverage(
        target_centroid=point, coverage_gdf=coverage_gdf, radius=circle_radius
    )
    describe_df = describe_random_network(
        network=network,
        target_centroid=point,
        name=name,
        amount_of_coverage=amount_of_coverage,
        radius=circle_radius,
    )

    return network, describe_df.iloc[0]


def describe_random_network(
    network: Optional[Network],
    target_centroid: Point,
    radius: float,
    name: str,
    amount_of_coverage: float,
) -> pd.DataFrame:
    """
    Describe a Network.

    Accepts None which results in numerical_network_description with all zero
    values.

    TODO: Zero to np.nan?
    """
    empty_numerical_network_description = empty_numerical_desc()
    if network is not None:
        numerical_network_description = network.numerical_network_description()
        # Validate empty_numerical_network_description assuming that atleast
        # some non-empty data is passed here
        # (hacky)
        assert all(
            [
                key in empty_numerical_network_description
                for key in numerical_network_description
            ]
        )
        trace_boundary_counts, branch_boundary_counts = (
            network.trace_boundary_intersect_count,
            network.branch_boundary_intersect_count,
        )
        assert trace_boundary_counts is not None
        assert branch_boundary_counts is not None
        numerical_network_description = {
            **numerical_network_description,
            **trace_boundary_counts,
            **branch_boundary_counts,
        }
    else:
        numerical_network_description = empty_numerical_network_description
    describe_df = create_describe_df(
        target_centroid=target_centroid,
        name=name,
        radius=radius,
        numerical_network_description=numerical_network_description,
        amount_of_coverage=amount_of_coverage,
        describe_df_schema=describe_df_schema,
    )
    assert not describe_df.empty
    return describe_df


def empty_numerical_desc():
    """
    Create empty numerical description.

    TODO: Not stable, keys are as strings and not validated.
    """
    keys = [
        "X",
        "Y",
        "I",
        "E",
        "C - C",
        "C - I",
        "I - I",
        "C - E",
        "I - E",
        "E - E",
        "trace power_law vs. lognormal R",
        "trace power_law vs. lognormal p",
        "trace power_law vs. exponential R",
        "trace power_law vs. exponential p",
        "trace lognormal vs. exponential R",
        "trace lognormal vs. exponential p",
        "trace power_law vs. truncated_power_law R",
        "trace power_law vs. truncated_power_law p",
        "trace power_law Kolmogorov-Smirnov distance D",
        "trace exponential Kolmogorov-Smirnov distance D",
        "trace lognormal Kolmogorov-Smirnov distance D",
        "trace truncated_power_law Kolmogorov-Smirnov distance D",
        "trace power_law alpha",
        "trace power_law exponent",
        "trace power_law cut-off",
        "trace power_law sigma",
        "trace lognormal sigma",
        "trace lognormal mu",
        "trace exponential lambda",
        "trace truncated_power_law lambda",
        "trace truncated_power_law alpha",
        "trace truncated_power_law exponent",
        "trace lognormal loglikelihood",
        "trace exponential loglikelihood",
        "trace truncated_power_law loglikelihood",
        "branch power_law vs. lognormal R",
        "branch power_law vs. lognormal p",
        "branch power_law vs. exponential R",
        "branch power_law vs. exponential p",
        "branch lognormal vs. exponential R",
        "branch lognormal vs. exponential p",
        "branch power_law vs. truncated_power_law R",
        "branch power_law vs. truncated_power_law p",
        "branch power_law Kolmogorov-Smirnov distance D",
        "branch exponential Kolmogorov-Smirnov distance D",
        "branch lognormal Kolmogorov-Smirnov distance D",
        "branch truncated_power_law Kolmogorov-Smirnov distance D",
        "branch power_law alpha",
        "branch power_law exponent",
        "branch power_law cut-off",
        "branch power_law sigma",
        "branch lognormal sigma",
        "branch lognormal mu",
        "branch exponential lambda",
        "branch truncated_power_law lambda",
        "branch truncated_power_law alpha",
        "branch truncated_power_law exponent",
        "branch lognormal loglikelihood",
        "branch exponential loglikelihood",
        "branch truncated_power_law loglikelihood",
        "Number of Traces",
        "Number of Branches",
        "Fracture Intensity B21",
        "Fracture Intensity P21",
        "Areal Frequency P20",
        "Areal Frequency B20",
        "Trace Mean Length",
        "Branch Mean Length",
        "Dimensionless Intensity P22",
        "Dimensionless Intensity B22",
        "Connections per Trace",
        "Connections per Branch",
        "Fracture Intensity (Mauldon)",
        "Fracture Density (Mauldon)",
        "Trace Mean Length (Mauldon)",
        "Connection Frequency",
        "trace lengths cut off proportion",
        "branch lengths cut off proportion",
    ]

    return {key: 0.0 for key in keys}


def save_describe_df(describe_df: pd.DataFrame, results_path: Path):
    """
    Save description DataFrame.
    """
    if results_path.exists():
        df = read_csv(results_path)
        assert isinstance(df, pd.DataFrame)
        concatted = pd.concat([df, describe_df])
    else:
        concatted = describe_df
    assert not concatted.empty

    save_csv(concatted, results_path)
    concatted.to_pickle(results_path.with_suffix(".pickle"))
