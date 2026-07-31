"""Packaging definition for project-iron.

This file exists so the repository can be installed with ``pip install -e .``
and imported as the ``src`` package from anywhere. Before this existed, every
entrypoint mutated the import path at runtime to make ``src/`` importable, which
made module identity depend on the current working directory: the same module
could be imported twice under two different names (``models.dav2_wrapper`` and
``src.models.dav2_wrapper``), each with its own class objects and global state.

Runtime dependencies are deliberately NOT declared here. They are pinned in
``locking-requirements.txt`` because several of them (torch, openvino) need
custom index URLs that ``install_requires`` cannot express.
"""

from setuptools import find_packages, setup

setup(
    name="project-iron",
    version="0.1.0",
    description=(
        "CPU-only video-intelligence pipeline: depth, point tracking, "
        "patch embeddings, and event extraction."
    ),
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    include_package_data=False,
)
