"""Setuptools shim for Tessrax package installation.

This setup script enforces Tessrax governance clauses by delegating
metadata to setuptools while ensuring the package name and interpreter
constraints remain aligned with installation expectations.
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

BASE_DIR = Path(__file__).resolve().parent

setup(
    name="tessrax",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    include_package_data=True,
    description="Tessrax governance framework",
    long_description=(BASE_DIR / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
)
