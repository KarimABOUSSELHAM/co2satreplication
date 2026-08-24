# src/co2sat/utils.py
from pathlib import Path


def project_root() -> Path:
    """Return the absolute path to the project root.

    Searches upward from the current working directory for a directory
    containing pyproject.toml. Works from notebooks, scripts, or anywhere
    inside the project tree.
    """
    for parent in [Path.cwd(), *Path.cwd().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml)")


def data_dir(*parts: str) -> Path:
    """Return a Path within the project's data/ directory.

    Example: data_dir("raw", "epa_daily") -> <root>/data/raw/epa_daily
    """
    return project_root().joinpath("data", *parts)
