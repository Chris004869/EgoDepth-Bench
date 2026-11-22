"""Resolve indexed release assets without following links outside the dataset."""
from pathlib import Path


def inside(root, relative):
    """Keep source, destination and existing symlinks within the release root."""
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe release-relative path: {relative}")
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Release path escapes through a symlink: {relative}")
    return path
