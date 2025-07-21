import datetime
import importlib
import importlib.resources as pkg_resources  # Load resource from package
import inspect
import os
import shutil
import sys
from pathlib import Path
from typing import Union
from uuid import uuid4

import geopandas as gpd

import hhnk_research_tools.logging as logging

logger = logging.get_logger(__name__)


def get_functions(cls, stringify=True):
    """Get a string with functions (methods) in a class."""
    funcs = [
        f".{i}" for i in dir(cls) if not i.startswith("__") and hasattr(inspect.getattr_static(cls, i), "__call__")
    ]
    if stringify:
        funcs = " ".join(funcs)
    return funcs


def get_variables(cls, stringify=True):
    """Get a string with variables (properties) in a class."""
    variables = [
        i for i in dir(cls) if not i.startswith("__") and not hasattr(inspect.getattr_static(cls, i), "__call__")
    ]
    if stringify:
        variables = " ".join(variables)
    return variables


def ensure_file_path(filepath):
    # TODO add to file class? still needed?
    """Make sure all folders in a given file path exist.
    Creates them if they don't.
    """
    try:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise e from None


def convert_gdb_to_gpkg(gdb, gpkg, overwrite=False, verbose=True):
    """Convert input filegdb to geopackage"""

    if gdb.exists():
        if check_create_new_file(output_file=gpkg, overwrite=overwrite):
            if verbose:
                print(f"Write gpkg to {gpkg}")
            for layer in gdb.available_layers():
                if verbose:
                    print(f"    {layer}")
                gdf = gpd.read_file(str(gdb), layer=layer)

                gdf.to_file(str(gpkg), layer=layer, driver="GPKG")


def check_create_new_file(
    output_file: Union[Path, str],
    overwrite: bool = False,
    input_files: Union[list[Path], None] = None,
    remove_when_older: bool = True,
    check_is_file=True,
) -> bool:
    """
    Check if we should continue to create a new file.

    Parameters
    ----------
    output_file : Path | str
        The file under evaluation
    overwrite : bool, default is True
        When overwriting is True the output_file will be removed.
    input_files : list[Path], default is None
        When input files are provided the edit time of the input will be
        compared to the output. If edit time is after the output, it will
        recreate the output.
    remove_when_older : bool, default is True
        There may be scenario's where this automatic removal of
        older output_files is unwanted. Changing this to False,
        will skip that step. Make sure the file is deleted / overwritten
        when create
    check_is_file bool, default is True
        When True, raises TypeError if output_file is not a file

    Returns
    -------
    create : bool,
        Bool if the inputfile should be created.
    """
    create = False
    output_file = Path(str(output_file))

    # Raise when no suffix if checking for file
    if check_is_file:
        if not output_file.suffix:
            raise TypeError(f"{output_file} is not a file.")

    # Check if we should create the file
    if not output_file.exists():
        create = True
    elif overwrite:
        logger.info(f"check_create_new_file - Removed {output_file.name} due to overwrite=True")
        output_file.unlink()
        create = True
    elif input_files:
        # Check edit times. To see if raster needs to be updated.
        output_mtime = output_file.stat().st_mtime

        for input_file in input_files:
            input_file = Path(str(input_file))
            if input_file.exists():
                input_mtime = input_file.stat().st_mtime

                if input_mtime > output_mtime:
                    if remove_when_older:
                        logger.info(
                            f"check_create_new_file - Removed {output_file.name} due to creation time older than input"
                        )
                        output_file.unlink()
                    else:
                        logger.info(
                            f"check_create_new_file - File {output_file.name} has creation time older than input. But was not removed."
                        )
                    create = True
                    break
    return create


def load_source(name: str, path: str):
    """Load python file as module.

    Replacement for deprecated imp.load_source()
    Inspiration from https://github.com/cuthbertLab/music21/blob/master/music21/test/commonTest.py
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"No such file or directory: {path!r}")
    if name in sys.modules:
        module = sys.modules[name]
    else:
        module = importlib.util.module_from_spec(spec)
        if module is None:
            raise FileNotFoundError(f"No such file or directory: {path!r}")
        sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def get_uuid(chars=8) -> str:
    """Max chars is 36"""
    return str(uuid4())[:chars]


def get_pkg_resource_path(package_resource, name) -> Path:
    """Return path to resource in a python package, so it can be loaded"""
    with pkg_resources.path(package_resource, name) as p:
        return p.absolute().resolve()


def current_time(time_format="%H:%M:%S", date: bool = False) -> str:
    if date is True:
        time_format = "%Y%m%d_%H%M%S_%f"
    return datetime.datetime.now().strftime(time_format)


def time_delta(start_time: datetime.datetime):
    """Difference between starttime in seconds.

    start_time (datetime.datetime): get by using datetime.datetime.now()
    """
    return round((datetime.datetime.now() - start_time).total_seconds(), 2)


def remove_cache_dirs(repo_path: os.PathLike) -> None:
    """Recusively remove cached files from repository.
    The cache dirs can grow quite a bit during development. This cleans up the folder.

    Removes all these folders in the main and its subfolders.
    .mypy_cache
    .pytest_cache
    .ruff_cache
    __pycache__

    Parameters
    ----------
    repo_path : os.PathLike
        Path to repository
    """
    repo_path = Path(repo_path)
    for name in [
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
    ]:
        for cachedir in repo_path.rglob(name):
            if cachedir.is_dir():
                if ".pixi" not in str(cachedir):
                    logger.info(f"Remove: {cachedir}")
                    shutil.rmtree(cachedir, ignore_errors=True)


class dict_to_class(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
