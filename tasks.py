"""
Invoke tasks.

Most tasks employ nox to create a virtual session for testing.
"""
from pathlib import Path
from shutil import copytree, rmtree

from invoke import task

nox_test_sessions = (
    "tests_pipenv",
    "tests_pip",
)

package_name = "fractopo_subsampling"


@task
def requirements(c):
    """
    Sync requirements from Pipfile to setup.py.
    """
    c.run("nox --session requirements")


@task
def sync_notebooks(_):
    """
    Sync notebooks from local development directory.
    """
    local_dev_notebooks_dir = Path("/mnt/d/Data/trace_repo/notebooks")

    notebooks_dir = Path("scripts_and_notebooks/notebooks")

    if not (local_dev_notebooks_dir.exists() and local_dev_notebooks_dir.is_dir()):
        print(f"No local dev dir found at {local_dev_notebooks_dir}.")
        return

    if notebooks_dir.exists():

        print(f"Removing {notebooks_dir}.")
        rmtree(notebooks_dir)

    print(f"Copying directory tree from {local_dev_notebooks_dir} to {notebooks_dir}.")
    copytree(local_dev_notebooks_dir, notebooks_dir)


@task(pre=[sync_notebooks])
def format(c):
    """
    Format everything.
    """
    c.run("nox --session format")


@task(pre=[format])
def lint(c):
    """
    Lint everything.
    """
    c.run("nox --session lint")


@task(pre=[requirements])
def nox_tests(c):
    """
    Run selected nox test suite sessions.
    """
    for nox_test in nox_test_sessions:
        c.run(
            f"nox --session {nox_test} --no-color",
        )


@task(pre=[sync_notebooks, requirements])
def ci_test(c):
    """
    Test suite for continous integration testing.

    Installs with pip, tests with pytest and checks coverage with coverage.
    """
    c.run("nox --session tests_pip")


@task(pre=[nox_tests])
def test(_):
    """
    Run tests.

    This is an extensive suite. It first tests in current environment and then
    creates virtual sessions with nox to test installation -> tests.
    """


@task(pre=[requirements])
def docs(c):
    """
    Make documentation to docs using nox.
    """
    c.run("nox --session docs")


@task(pre=[sync_notebooks, requirements, test, lint, docs])
def make(_):
    """
    Make all.
    """
    print("---------------")
    print("make successful.")
