"""
Command line interface of subsampling scripts.
"""
import logging
from pathlib import Path

import click
from fractopo.analysis.random_sampling import NetworkRandomSampler
from fractopo.general import read_geofile

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
@click.option("how_many", "--how-many", type=click.IntRange(1, 100), default=1)
@click.option("hashname", "--hashname", is_flag=True, default=False)
def subsample(
    traces_path_str: str,
    area_path_str: str,
    results_path_str: str,
    other_results_path_str: str,
    coverage_path_str: str,
    how_many: int,
    hashname: bool,
):
    """
    Conduct single network subsampling within the given sample area.
    """
    traces_path = Path(traces_path_str)
    area_path = Path(area_path_str)
    other_results_path = Path(other_results_path_str)
    results_path = Path(results_path_str)
    coverage_path = Path(coverage_path_str)
    trace_gdf = read_geofile(traces_path)
    area_gdf = read_geofile(area_path)
    coverage_gdf = read_geofile(coverage_path)
    sampler = NetworkRandomSampler(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        min_radius=5,
        snap_threshold=0.001,
        random_choice="radius",
    )

    for _ in range(how_many):

        network, target_centroid, radius = sampler.random_network_sample()
        amount_of_coverage = assess_coverage(target_centroid, radius, coverage_gdf)
        if hashname:
            name_hash = abs(hash(target_centroid.wkt))
            results_path = (
                results_path.parent / f"{results_path.stem}_{name_hash}"
                f"{results_path.suffix}"
            )
            if results_path.exists():
                more_complex_hash = abs(hash(target_centroid.wkt) + hash(radius))
                results_path = (
                    results_path.parent / f"{results_path.stem}_{more_complex_hash}"
                    f"{results_path.suffix}"
                )
                assert not results_path.exists()

        describe_df = describe_random_network(
            network=network,
            target_centroid=target_centroid,
            radius=radius,
            name=area_path.stem,
            amount_of_coverage=amount_of_coverage,
        )

        save_describe_df(describe_df, results_path=results_path)
        if network is not None:
            save_azimuth_bin_data(
                network,
                other_results_path,
                loc_hash=f"{area_path.stem}_{target_centroid.wkt}_{radius}",
            )


@main.command()
@click.argument("results_path_str", type=click.Path(exists=True, dir_okay=True))
@click.argument("gather_path_str", type=click.Path(dir_okay=False))
def gather_subsampling(results_path_str: str, gather_path_str: str):
    """
    Gather subsampling results.
    """
    results_path = Path(results_path_str)
    gather_path = Path(gather_path_str)
    concatted = gather_subsampling_results(results_path=results_path)
    save_csv(concatted, gather_path)
    concatted.to_pickle(gather_path.with_suffix(".pickle"))
    print(f"Saved concatted data at {gather_path}.")
