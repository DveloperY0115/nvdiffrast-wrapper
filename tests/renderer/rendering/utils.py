"""
utils.py

A collection of utilities used for testing renderer.
"""

from pathlib import Path

from jaxtyping import jaxtyped
import numpy as np
import torch
from typeguard import typechecked


@jaxtyped
@typechecked
def load_cube(file: Path):
    """Loads a cube used for testing"""

    # assertions
    assert file.exists(), (
        f"No such file: {str(file)}"
    )

    with np.load(str(file)) as filestream:
        (
            faces,
            vertices,
            _,
            vertex_colors,
        ) = filestream.values()

    faces = torch.tensor(faces.astype(np.int32))
    vertices = torch.tensor(vertices, dtype=torch.float32)
    vertex_colors = torch.tensor(vertex_colors, dtype=torch.float32)

    return faces, vertices, vertex_colors
