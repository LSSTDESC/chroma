"""
Functions to find data files.
"""

import importlib.resources


def find_data():
    """
    Find the path to the data directory.
    """
    data = importlib.resources.files("chroma") / "data"
    return data


def find_filter(path):
    data = find_data()
    p = data / "filters" / path
    assert p in filters(), f"{path} not found in {filters()}"
    return p.as_posix()


def find_SED(path):
    data = find_data()
    p = data / "SEDs" / path
    assert p in SEDs(), f"{path} not found in {SEDs()}"
    return p.as_posix()


def find_simard(path):
    data = find_data()
    p = data / "simard" / path
    assert p in simards(), f"{path} not found in {simards()}"
    return p.as_posix()


def filters():
    data = find_data()
    return {f for f in (data / "filters").glob("*.dat")}


def SEDs():
    data = find_data()
    return (
        {f for f in (data / "SEDs").glob("*.ascii")}
        | {f for f in (data / "SEDs").glob("*.spec")}
        | {f for f in (data / "SEDs").glob("*.dat")}
    )


def simards():
    data = find_data()
    return {f for f in (data / "SEDs").glob("*.fits")}
