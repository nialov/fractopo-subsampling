"""
Command line interface of subsampling scripts.
"""
import logging
from pathlib import Path
from typing import List, Sequence

import click
from fractopo.analysis.random_sampling import NetworkRandomSampler
from fractopo.general import read_geofile

import fractopo_subsampling.utils as utils
from fractopo_subsampling.network_scripts import (
    analyze,
    assess_coverage,
    describe_random_network,
    gather_results,
    gather_subsampling_results,
    save_azimuth_bin_data,
    save_csv,
    save_describe_df,
    save_results,
)


@click.group()
def main():
    """
    Fractopo network sampling interface.
    """
    pass


@main.command()
@click.argument("results_path_str", type=click.Path(exists=True, dir_okay=True))
@click.argument("save_path_str", type=click.Path(dir_okay=False))
def gatherbase(results_path_str: str, save_path_str: str):
    """
    Gather run results to GeoPackage.
    """
    logging.debug(f"Gathering from {results_path_str} into {save_path_str}.")
    results_path = Path(results_path_str)
    save_path = Path(save_path_str)
    gdf = gather_results(results_path)
    logging.debug("GeoDataFrame made. Saving.")
    save_results(gdf, save_path=save_path)


@main.command()
@click.argument("traces_path_str", type=click.Path(exists=True, dir_okay=False))
@click.argument("area_path_str", type=click.Path(exists=True, dir_okay=False))
@click.argument("results_path_str", type=click.Path(exists=True, dir_okay=True))
@click.argument("other_results_path_str", type=click.Path(exists=True, dir_okay=True))
@click.argument("coverage_path_str", type=click.Path(exists=True, dir_okay=False))
@click.argument("circle_radius", type=click.FloatRange(min=0.001))
@click.option("overwrite", "--overwrite", is_flag=True, default=False)
def baseanalyze(
    traces_path_str: str,
    area_path_str: str,
    results_path_str: str,
    other_results_path_str: str,
    coverage_path_str: str,
    circle_radius: float,
    overwrite: bool,
) -> bool:
    """
    Run individual network analyses.
    """
    traces_path = Path(traces_path_str)
    area_path = Path(area_path_str)
    results_path = Path(results_path_str)
    other_results_path = Path(other_results_path_str)
    coverage_path = Path(coverage_path_str)

    if not (results_path.exists() and results_path.is_dir()):
        raise NotADirectoryError(f"Expected {results_path} dir to exist.")

    traces, area = read_geofile(traces_path), read_geofile(area_path)
    coverage_gdf = read_geofile(coverage_path)
    name = area_path.stem
    result_path = results_path / f"{name}.pickle"
    if result_path.exists() and not overwrite:
        return False
    description_srs = analyze(
        traces,
        area,
        name,
        other_results_path,
        coverage_gdf=coverage_gdf,
        circle_radius=circle_radius,
    )
    if result_path.exists():
        result_path.unlink()
    description_srs.to_pickle(results_path / f"{name}.pickle")

    return True


@main.command()
@click.argument("traces_path_str", type=click.Path(exists=True, dir_okay=False))
@click.argument("area_path_str", type=click.Path(exists=True, dir_okay=False))
@click.argument("results_path_str", type=click.Path(dir_okay=False))
@click.argument("other_results_path_str", type=click.Path(exists=True, dir_okay=True))
@click.argument("coverage_path_str", type=click.Path(exists=True, dir_okay=False))
def subsample(
    traces_path_str: str,
    area_path_str: str,
    results_path_str: str,
    other_results_path_str: str,
    coverage_path_str: str,
):
    """
    Conduct single network subsampling within the given sample area.
    """
    # Convert to Paths
    traces_path = Path(traces_path_str)
    area_path = Path(area_path_str)
    other_results_path = Path(other_results_path_str)
    results_path = Path(results_path_str)
    coverage_path = Path(coverage_path_str)

    # Read GeoDataFrames
    trace_gdf = read_geofile(traces_path)
    area_gdf = read_geofile(area_path)
    coverage_gdf = read_geofile(coverage_path)

    # Initialize NetworkRandomSampler
    sampler = NetworkRandomSampler(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        min_radius=5,
        snap_threshold=0.001,
        random_choice="radius",
    )

    # Create random network sample
    # Returns fractopo Network instance, centroid shapely Point and
    # radius of the sample circle
    network, target_centroid, radius = sampler.random_network_sample()

    # Assess the amount of censoring within the sample
    amount_of_coverage = assess_coverage(target_centroid, radius, coverage_gdf)

    # Use the sample centroid Point and hash its wkt string repr
    name_hash = abs(hash(target_centroid.wkt))

    # Resolve path
    save_path = (
        results_path.parent / f"{results_path.stem}_{name_hash}"
        f"{results_path.suffix}"
    )

    # If there's hash conflict, make more complex hash
    if results_path.exists():
        more_complex_hash = abs(hash(target_centroid.wkt) + hash(radius))
        save_path = (
            results_path.parent / f"{results_path.stem}_{more_complex_hash}"
            f"{results_path.suffix}"
        )
        assert not save_path.exists()

    # Resolve the Network instance to just a dataframe of the parameters
    # we are interested in
    describe_df = describe_random_network(
        network=network,
        target_centroid=target_centroid,
        radius=radius,
        name=area_path.stem,
        amount_of_coverage=amount_of_coverage,
    )

    # Save the dataframe
    save_describe_df(describe_df, results_path=save_path)

    # Save azimuth rose plots
    if network is not None:
        save_azimuth_bin_data(
            network,
            other_results_path,
            loc_hash=f"{area_path.stem}_{target_centroid.wkt}_{radius}",
        )


@main.command()
@click.argument("results_path_str", type=click.Path(exists=True, dir_okay=True))
@click.argument("gather_path_str", type=click.Path(dir_okay=False))
def gather_subsamples(results_path_str: str, gather_path_str: str):
    """
    Gather subsampling results.
    """
    results_path = Path(results_path_str)
    gather_path = Path(gather_path_str)
    concatted = gather_subsampling_results(results_path=results_path)
    save_csv(concatted, gather_path)
    concatted.to_pickle(gather_path.with_suffix(".pickle"))
    print(f"Saved concatted data at {gather_path}.")


@main.command()
@click.option(
    "csv_name",
    "--csv-name",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "censoring_geopackage",
    "--censoring-geopackage",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "results_dir",
    "--results-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
)
@click.option(
    "azimuth_results_dir",
    "--azimuth-results-dir",
    type=click.Path(exists=True, dir_okay=True),
    required=True,
)
@click.option(
    "how_many",
    "--how-many",
    type=click.IntRange(min=1, max=10000),
    required=True,
)
@click.option(
    "filter_pattern", "--filter-pattern", type=click.STRING, default="", required=False
)
def network_subsampling(
    csv_name: str,
    censoring_geopackage: str,
    results_dir: str,
    azimuth_results_dir: str,
    filter_pattern: str,
    how_many: int,
):
    """
    Run network subsampling on all valid areas in parallel.

    Calls subsample cli command in parallel.
    """
    # Resolve paths
    traces_paths, area_paths, marks = utils.collect_paths(
        csv_name, skip=[utils.Skip.invalid, utils.Skip.empty], filter=filter_pattern
    )

    # Collect a list of command sequences
    cmds: List[Sequence[str]] = []

    # Iterate over each path
    for trace_path, area_path, _ in zip(traces_paths, area_paths, marks):

        # Resolve results path
        results_path = Path(results_dir) / f"{area_path.stem}.csv"

        # Resolve path for misc results
        other_results_path = Path(azimuth_results_dir)

        # Create command tuple
        cmd_raw = (
            *utils.SCRIPTS_RUN_CMD.split(" "),
            "subsample",
            trace_path,
            area_path,
            results_path,
            other_results_path,
            censoring_geopackage,
        )

        # Convert all tuple values to string and save to list
        cmd_str = list(map(str, cmd_raw))

        # Add command sequence to list how_many times
        for _ in range(how_many):
            cmds.append(cmd_str)

    # Run all command sequences in parallel
    utils.async_run(cmds)
