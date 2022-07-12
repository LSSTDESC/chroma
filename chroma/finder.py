"""
Functions to find data files.
"""

import importlib.resources
from importlib.abc import Traversable


def find_data():
    """
    Find the path to the data directory.
    """
    data = importlib.resources.files("chroma") / "data"
    return data


def find_filter(path: str) -> str:
    data = find_data()
    p = data / "filters" / path
    assert p in filters(), f"{path} not found in {filters()}"
    return p.as_posix()


def find_SED(path: str) -> str:
    data = find_data()
    p = data / "SEDs" / path
    assert p in SEDs(), f"{path} not found in {SEDs()}"
    return p.as_posix()


def find_simard(path: str) -> str:
    data = find_data()
    p = data / "simard" / path
    assert p in simards(), f"{path} not found in {simards()}"
    return p.as_posix()


def filters() -> set:
    data = find_data()
    return {f for f in (data / "filters").glob("*.dat")}


def SEDs() -> set:
    data = find_data()
    return (
        {f for f in (data / "SEDs").glob("*.ascii")}
        | {f for f in (data / "SEDs").glob("*.spec")}
        | {f for f in (data / "SEDs").glob("*.dat")}
    )


def simards() -> set:
    data = find_data()
    return {f for f in (data / "SEDs").glob("*.fits")}
