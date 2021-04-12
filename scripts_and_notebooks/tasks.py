"""
Invoke task invocations.

This is a mixed collection of scripts related to handling the file and
directory structure within the working folder and calls to fractopo-subsampling.
cli functions. It is not pretty but it works.

Better file management and validation framework is a work-in-progress.
"""

import csv
from pathlib import Path
from typing import List, Literal, Tuple
from click import confirm
from shutil import copytree, rmtree

from invoke import UnexpectedExit, call, task
import time
from invoke.exceptions import Exit
from invoke.runners import Result
from enum import Enum, unique
import geopandas as gpd

scripts_run_cmd = "python3 -m fractopo_subsampling"


@unique
class Skip(Enum):

    """
    Enums for skip options.
    """

    valid = "valid"
    invalid = "invalid"
    empty = "empty"


# Base results
RESULTS_DIR = Path("results")

# Base notebooks dir
NOTEBOOKS_DIR = Path("notebooks")

# Dir for base circle analysis pickles
PICKLES_DIR = RESULTS_DIR / "base_circle_pickles"

# Dir for analyzed base circle notebooks
ANALYZED_NOTEBOOKS = RESULTS_DIR / "analyzed_base_circle_notebooks"

# Dir for miscellanous
MISC_DIR = RESULTS_DIR / "misc"

# Subsampling results base dir
SUBSAMPLING_RESULTS_DIR = RESULTS_DIR / "subsampling"

# Azimuth rose plots are collected
AZIMUTH_SUBSAMPLING_RESULTS_DIR = SUBSAMPLING_RESULTS_DIR / "azimuths"

# Subsampling result aggregation csvs to collected
COLLECTED_SUBSAMPLING_RESULTS_DIR = SUBSAMPLING_RESULTS_DIR / "collected"

# Base circle analyzed points with parameters
ANALYSIS_GPKG = RESULTS_DIR / "Ahvenanmaa_analysis_points.gpkg"

# Base network for papermill parametrization
BASE_NETWORK_NB = NOTEBOOKS_DIR / "fractopo_network.ipynb"

# organize-tool config file
BASE_ORGANIZE_FILE = "organize.yaml"

# csv with target area and trace links and metadata
RELATIONS_CSV = "relations.csv"

# GeoPackage with censored areas digitized as polygons
CENSORING_PATH = Path("censoring_in_target_areas.gpkg")

# For directory shapefile conversion output
SHP_PATH = Path("ahvenanmaa_shp")


def validate_relations_csv_rows(reader):
    """
    Validate rows in relations csv.
    """
    # Read headers from template_file
    relations_csv_contents = (
        (Path(__file__).parent / RELATIONS_CSV).read_text().split("\n")[0].strip()
    )

    # Get headers in csv
    headers = relations_csv_contents.split(",")
    row_length = len(headers)
    assert row_length == 7

    names = []
    for idx, row in enumerate(reader):

        if idx == 0 and not all([header in row for header in headers]):
            # Check headers
            raise ValueError(f"Expected headers: {headers} in relations.csv.")

        if not len(row) == row_length:
            raise ValueError(f"Row idx {idx} was not of length {row_length}")
        if not all([isinstance(item, str) and len(item) > 0 for item in row]):
            raise ValueError(f"Row idx {idx} did not contain valid items.")
        if any([" " in item for item in row]):
            raise ValueError(f"Row idx {idx} contained spaces in item.")

        names.append(row[0])

    if not len(names) > 0:
        raise ValueError("Expected relations.csv to contain headers.")

    dupes = set()
    for name in names:
        if name not in dupes and names.count(name) > 1:
            dupes.add(name)
    if len(dupes) > 0:
        raise ValueError(f"Duplicate names: {dupes}")


def validate_relations_csv(csv_name: str = RELATIONS_CSV):
    """
    Validate given relations csv.

    Defaults to RELATIONS_CSV variable.
    """
    csv_path = Path(csv_name)
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_name} to exist as file.")

    with csv_path.open("r") as csvfile:
        reader = csv.reader(csvfile)
        validate_relations_csv_rows(reader)


def run_organize(c, simulate=True, confirm_overwrite=False):
    """
    Organize trace and area data.

    Simulates by default i.e. no real changes applied to file system.
    """
    command_base = "organize {} --config={}"
    if not Path(BASE_ORGANIZE_FILE).exists():
        raise FileNotFoundError(f"Expected {BASE_ORGANIZE_FILE} file to exist.")

    # Whether to simulate or run for real
    sim_or_run = "sim" if simulate else "run"

    # Categorize traces and areas by geometry
    base_result = c.run(command_base.format(sim_or_run, BASE_ORGANIZE_FILE))

    if "ERROR!" in str(base_result.stdout):
        # Error in simulation or running
        # raise Exception("Error in organize.")
        raise Exit(message="Error in organize.")
    if (
        "File already exists" in str(base_result.stdout)
        and simulate
        and confirm_overwrite
    ):
        if not confirm("Overwrite already organized files?"):
            raise Exit(message="Overwriting was not allowed by user.")


def collect_paths(
    csv_path,
    skip: List[Literal[Skip.empty, Skip.valid, Skip.invalid]],
    filter: str = "",
) -> Tuple[List[Path], List[Path], List[Tuple[str, str, float]]]:
    """
    Collect trace and area paths from relations.csv file.
    """
    if not all([val in Skip for val in skip]):
        raise ValueError(f"Expected skip vals to be one of {Skip}.")
    traces_paths, area_paths, marks = [], [], []
    with Path(csv_path).open("r") as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            base_path = "{}/{}/{}/{}"
            area_path = Path(
                base_path.format(row[2], "areas", row[3], row[0] + ".gpkg")
            ).resolve()
            traces_path = Path(
                base_path.format(row[2], "traces", row[3], row[1] + ".gpkg")
            ).resolve()
            if not (area_path.exists() and traces_path.exists()):
                continue
            if row[4] == "True" and Skip.valid in skip:
                # Do not collect valid.
                continue
            if row[4] == "False" and Skip.invalid in skip:
                # Do not collect invalid.
                continue
            if row[5] == "True" and Skip.empty in skip:
                # Do not collect empty.
                continue
            if len(filter) > 0 and filter not in area_path.stem:
                # Only collect area paths that fit filter.
                continue

            traces_paths.append(traces_path)
            area_paths.append(area_path)
            # valid, empty, radius
            diameter = float(row[-1])
            marks.append((row[4], row[5], diameter))
    return traces_paths, area_paths, marks


def promise_handler(
    promise, promises, promise_count, incremental_results
) -> Tuple[list, int, list]:
    """
    Handle promises so that only a set amount is executed at once.
    """
    promises.append(promise)
    promise_count += 1
    promise_count_threshold = 24
    incremental_results, promise_count, promises = promise_queue(
        promise_count=promise_count,
        promise_count_threshold=promise_count_threshold,
        incremental_results=incremental_results,
        promises=promises,
    )
    return promises, promise_count, incremental_results


def promise_queue(
    promise_count: int,
    promise_count_threshold: int,
    incremental_results: list,
    promises: list,
) -> Tuple[list, int, list]:
    """
    Queue tasks instead of all at once.
    """
    if promise_count > 24:
        print(f"Reached {promise_count_threshold} tasks. Waiting on promises.")
        some_results, promise_count, promises = join_some_promises(
            promises, promise_count, threshold=10
        )
        promise_count = len(promises)
        incremental_results.extend(some_results)
        print(f"Finished total of {len(incremental_results)} tasks. Continuing.")
    return incremental_results, promise_count, promises


def join_some_promises(promises: list, promise_count: int, threshold: int):
    """
    Join promises until threshold.
    """
    assert promise_count == len(promises)
    start_length = len(promises)
    assert promise_count > threshold
    some_results = []
    for promise in promises:
        if promise_count < threshold:
            promises_left = promises[len(some_results) :]
            assert start_length == len(promises_left) + len(some_results)
            return some_results, promise_count, promises_left
        some_results.append(promise.join())
        promise_count -= 1
    raise ValueError("Expected loop to break.")


def curr_datetime() -> str:
    """
    Get current time as formatted string.

    >>> curr_datetime()
    '16022021'

    """
    localtime = time.localtime()
    return time.strftime("%d%m%Y", localtime)


@task
def directories(_):
    """
    Check and make directory structure.
    """
    for path in (
        RESULTS_DIR,
        PICKLES_DIR,
        ANALYZED_NOTEBOOKS,
        MISC_DIR,
        SUBSAMPLING_RESULTS_DIR,
        AZIMUTH_SUBSAMPLING_RESULTS_DIR,
        COLLECTED_SUBSAMPLING_RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


@task
def check_csv(_):
    """
    Validate relations.csv file.

    Will error out if items and rows are not valid with informative error
    message.
    """
    validate_relations_csv()


@task(pre=[check_csv])
def simulate(c, confirm_overwrite=False):
    """
    Run organize command simulation.

    Errors if some file is not categorizable.
    """
    run_organize(c, simulate=True, confirm_overwrite=confirm_overwrite)


@task(pre=[call(simulate, confirm_overwrite=True)])
def organize(c):
    """
    Run organize command.

    Checks that all files are categorizable with simulation first.  Simulation
    can error in which case this will not run.  Simulation will also prompt
    user for input if overwriting is about to occur.
    """
    run_organize(c, simulate=False, confirm_overwrite=False)


@task
def check_for_empty(_):
    """
    Check for empty areas.

    Checks that all empty areas (no digitized traces within) are marked as such.
    """
    traces_paths, area_paths, marks = collect_paths(RELATIONS_CSV, skip=[])
    for traces_path, area_path in zip(traces_paths, area_paths):
        if traces_path.exists() and area_path.exists():
            trace_count = gpd.read_file(
                traces_path, mask=gpd.read_file(area_path)
            ).shape[0]
            if marks[1] == "True" and trace_count > 0:
                # marked as empty
                print(f"{area_path.stem} marked as empty but has {trace_count} rows.")
            if marks[1] == "False" and trace_count == 0:
                # marked as not empty
                print(
                    f"{area_path.stem} not marked as empty but has {trace_count} rows."
                )


@task(pre=[check_for_empty, check_csv])
def validate_all(c, only_area_validation=False, stderr=False):
    """
    Call tracevalidate on all target areas.

    Will skip areas that are marked as valid in relations.csv.
    """
    try:
        c.run("tracevalidate --help", hide="both")
    except UnexpectedExit:
        print("Error: Expected tracevalidate to be a defined shell command.")
        print("tracevalidate is a command line script in fractopo.")
        return

    traces_paths, area_paths, _ = collect_paths(
        RELATIONS_CSV, skip=[Skip.valid, Skip.empty]
    )
    # Collect promises
    promises = []
    # Collect validated trace paths
    validated = []
    not_skipped_areas = []

    for traces_path, area_path in zip(traces_paths, area_paths):
        # Check if same traces are validated multiple times
        # If yes -> Only validate once. But the traces should be validated for
        # each target area.
        if not (traces_path.exists() and area_path.exists()):
            print(f"No data for {area_path.stem}")
            continue
        if traces_paths.count(traces_path) > 1 and traces_path in validated:
            print(
                f"Skipping duplicate validation for "
                f"{area_path.name} traces {traces_path.name}."
            )
            continue
        only_area_validation_str = (
            " --only-area-validation" if only_area_validation else ""
        )
        promises.append(
            c.run(
                f"tracevalidate {traces_path} {area_path}"
                + " --fix --snap-threshold 0.01"
                + f" --output {traces_path}"
                + " --summary"
                + " --no-empty-area"
                + only_area_validation_str,
                asynchronous=True,
            )
        )
        # Add the traces and areas that are actually validated
        validated.append(traces_path)
        not_skipped_areas.append(area_path)
    assert len(validated) == len(promises)
    if len(promises) == 0:
        print("Nothing to validate.")
        return

    results = [promise.join() for promise in promises]
    for result, traces, area in zip(results, validated, not_skipped_areas):
        if isinstance(result, Result):
            print("==========================")
            print(f"-  Stdout of area {area.name}/{traces.name} tracevalidate:")
            print(result.stdout)
            if stderr:
                print("-  Stderr:")
                print(result.stderr)
            print("==========================")
        else:
            print("==========================")
            print(f"-  Exception for area {area.name}/{traces.name} tracevalidate:")
            print(result)
            print("==========================")


@task
def clean_notebooks(_, filter=""):
    """
    Clean analyzed notebooks.
    """
    for notebook in ANALYZED_NOTEBOOKS.glob("*.ipynb"):
        if len(filter) > 0 and filter not in notebook.stem:
            print(f"Not cleaning {notebook.stem}")
            continue
        notebook.unlink()


@task
def lint_notebooks(c):
    """
    Lint base notebook(s) with flake8-nb.
    """
    c.run(
        "flake8_nb notebooks/fractopo_network.ipynb --max-line-length 120 --ignore E402"
    )


@task
def format_notebooks(c):
    """
    Format base notebooks in ./notebooks.
    """
    c.run("black-nb notebooks/*.ipynb")


@task(
    pre=[
        check_csv,
        format_notebooks,
        lint_notebooks,
        directories,
    ]
)
def network_all(
    c, stdout=False, filter="", notebooks=False, points=False, overwrite=False
):
    """
    Run fractopo network analysis individually for all areas.

    Uses papermill asynchronously to run all notebook analysis based on
    relations.csv.  If a filter is passed only target areas matching the filter
    will be analyzed.

    Uses a python script from fractopo_subsampling to get numerical
    description of networks for all areas.

    E.g. if filter='Getaberget' then 'Getaberget_1', 'Getaberget_2_2', etc.
    will be analyzed but not 'Segelskar_1'.
    """
    promises = []
    for trace_path, area_path, marks in zip(
        *collect_paths(RELATIONS_CSV, skip=[Skip.invalid, Skip.empty])
    ):
        if notebooks:
            if not BASE_NETWORK_NB.exists():
                raise FileNotFoundError(f"Expected {BASE_NETWORK_NB} to exist.")

            name = area_path.stem
            output_path = ANALYZED_NOTEBOOKS / f"{BASE_NETWORK_NB.stem}_{name}.ipynb"

            if not ANALYZED_NOTEBOOKS.exists():
                raise FileNotFoundError(f"Expected {ANALYZED_NOTEBOOKS} to exist.")

            if len(filter) > 0 and filter not in name:
                print(f"Filtering OUT {name}")
                continue
            if output_path.exists() and overwrite:
                output_path.unlink()
            elif output_path.exists():
                print(f"Not overwriting {output_path}.")
                continue
            if len(promises) % 8 == 0:
                time.sleep(20)
            promises.append(
                c.run(
                    f"papermill {BASE_NETWORK_NB} {output_path}"
                    f" -p trace_data {trace_path} -p area_data {area_path}"
                    f" -p name {name} -p circular_target_area True --start-timeout 360",
                    asynchronous=True,
                )
            )
        overwrite_option = "--overwrite" if overwrite else ""
        if points:
            promises.append(
                c.run(
                    f"{scripts_run_cmd} baseanalyze"
                    f" {trace_path} {area_path} {PICKLES_DIR} {MISC_DIR}"
                    f" {CENSORING_PATH} {marks[2] / 2} {overwrite_option}",
                    asynchronous=True,
                )
            )
    if len(promises) == 0:
        print("Nothing to analyze.")
        return
    results = [promise.join() for promise in promises]
    if stdout:
        [print(result) for result in results]

    if points:
        c.run(f"{scripts_run_cmd} gatherbase" f" {PICKLES_DIR} {ANALYSIS_GPKG}")
        print(f"Gathered base analyze data at {ANALYSIS_GPKG}")


@task(pre=[directories, simulate])
def network_subsampling(c, filter_pattern="", how_many=1):
    """
    Run network subsampling on all valid areas.

    Calls fractopo-subsampling network-subsampling cli entrypoint
    which handles asynchronous subsampling execution.

    Use filter_pattern to only subsample areas matching the filter.
    The amount of subsamples for each area is given with how_many.
    """
    cmd_list = [
        "python",
        "-m",
        "fractopo_subsampling",
        "network-subsampling",
        "--csv-name",
        RELATIONS_CSV,
        "--censoring-geopackage",
        CENSORING_PATH,
        "--results-dir",
        SUBSAMPLING_RESULTS_DIR,
        "--azimuth-results-dir",
        AZIMUTH_SUBSAMPLING_RESULTS_DIR,
        "--how-many",
        how_many,
    ]
    if len(filter_pattern) > 0:
        cmd_list.extend(("--filter-pattern", filter_pattern))

    c.run(" ".join(map(str, cmd_list)))


@task
def clean_subsamples(_):
    """
    Clean all subsampling results.

    Does not remove collected datasets that are named after date in `collected`.
    """
    for path in list(SUBSAMPLING_RESULTS_DIR.glob("*")) + list(
        AZIMUTH_SUBSAMPLING_RESULTS_DIR.glob("*")
    ):
        if path.is_file():
            path.unlink()


@task(pre=[directories])
def gather_subsamples(c):
    """
    Gather subsampling results into COLLECTED_SUBSAMPLING_RESULTS_DIR.
    """
    gather_path = (
        COLLECTED_SUBSAMPLING_RESULTS_DIR / f"Subsampling_results_{curr_datetime()}.csv"
    )
    c.run(
        f"{scripts_run_cmd} gather-subsamples"
        f" {SUBSAMPLING_RESULTS_DIR} {gather_path}"
    )


@task
def transform_to_shp(c):
    """
    Transform gpkg to shp.
    """
    if SHP_PATH.exists():
        rmtree(SHP_PATH)
    copytree("ahvenanmaa", SHP_PATH)
    for gpkg in SHP_PATH.rglob("*.gpkg"):
        output_name = gpkg.with_suffix(".shp")
        c.run(f"geotrans {gpkg} --to_type shp --output {output_name}")
        gpkg.unlink()
