"""Shared file-path heuristics used across ingestion phases."""

from __future__ import annotations


def is_test_file(file_path: str) -> bool:
    """Return True if *file_path* belongs to a test directory or file."""
    parts = file_path.replace("\\", "/").split("/")
    return (
        "tests" in parts
        or "test" in parts
        or any(p.startswith("test_") for p in parts)
        or file_path.endswith("conftest.py")
    )


def is_alembic_migration(file_path: str) -> bool:
    """Return True if *file_path* looks like an Alembic migration file.

    Requires a ``versions`` segment preceded by ``alembic`` or ``migrations``
    to avoid false positives on unrelated directories.
    """
    normalized = file_path.replace("\\", "/")
    if not normalized.endswith(".py"):
        return False
    parts = normalized.split("/")
    for i in range(1, len(parts)):
        if parts[i] == "versions" and parts[i - 1] in ("alembic", "migrations"):
            return True
    return False
