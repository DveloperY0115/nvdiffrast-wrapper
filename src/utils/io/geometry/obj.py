"""
obj.py

Utility functions for handling I/O of Wavefront OBJ files.
"""

from pathlib import Path
from typing import List, Tuple, Union

import imageio
from jaxtyping import Float, jaxtyped
import numpy as np
import torch
from torch import Tensor
from typeguard import typechecked

from src.geometry.mesh import Mesh

@jaxtyped
@typechecked
def load_obj(
    path: Path,
    device: torch.device,
) -> Mesh:
    """
    Loads an .obj file.
    """

    assert path.suffix == ".obj", f"Not an .obj file. Got {path.suffix}"
    assert path.exists(), f"File not found: {str(path)}"

    vertices = []
    faces = []
    tex_coordinates = []
    vertex_normals = []
    tex_coordinate_indices = []
    vertex_normal_indices = []
    texture_image = None

    with open(path, "r") as file:

        for line in file.readlines():

            # parse vertex coordinates
            if line.startswith("v "):
                vertices_ = _parse_vertex(line)
                vertices.append(vertices_)

            # parse faces
            elif line.startswith("f "):
                (
                    face_indices_,
                    tex_coord_indices_,
                    vertex_normal_indices_
                ) = _parse_face(line)
                faces.append(face_indices_)
                tex_coordinate_indices.append(tex_coord_indices_)
                vertex_normal_indices.append(vertex_normal_indices_)

            # parse texture coordinates
            elif line.startswith("vt "):
                tex_coordinates_ = _parse_tex_coordinates(line)
                tex_coordinates.append(tex_coordinates_)

            # parse vertex normals
            elif line.startswith("vn "):
                vertex_normals_ = _parse_vertex_normal(line)
                vertex_normals.append(vertex_normals_)

            else:
                pass  # ignore

    vertices = torch.tensor(vertices, dtype=torch.float32)
    faces = torch.tensor(faces, dtype=torch.int)

    if len(tex_coordinates) > 0:
        tex_coordinates = torch.tensor(
            tex_coordinates,
            dtype=torch.float32,
        )
    else:
        tex_coordinates = None

    if len(vertex_normals) > 0:
        vertex_normals = torch.tensor(
            vertex_normals,
            dtype=torch.float32,
        )
    else:
        vertex_normals = None

    if tex_coordinate_indices[0][0] is not None:
        tex_coordinate_indices = torch.tensor(
            tex_coordinate_indices,
            dtype=torch.int,
        )
    else:
        tex_coordinate_indices = None

    if vertex_normal_indices[0][0] is not None:
        vertex_normal_indices = torch.tensor(
            vertex_normal_indices,
            dtype=torch.int,
        )
    else:
        vertex_normal_indices = None

    if tex_coordinates is not None:
        texture_image = _load_texture_image(path)
        if texture_image is None:
            tex_coordinates = None
            tex_coordinate_indices = None

    mesh = Mesh(
        vertices=vertices,
        faces=faces,
        tex_coordinates=tex_coordinates,
        vertex_normals=vertex_normals,
        tex_coordinate_indices=tex_coordinate_indices,
        vertex_normal_indices=vertex_normal_indices,
        texture_image=texture_image,
        has_texture=texture_image is not None,
        has_normal=vertex_normals is not None,
        device=device,
    )

    return mesh

@typechecked
def _parse_vertex(line: str) -> List[float]:
    return [float(x) for x in line.split()[1:]]

@typechecked
def _parse_face(
    line: str,
) -> Tuple[
    List[int],
    Union[List[int], List[None]],
    Union[List[int], List[None]],
]:
    """
    Parses a line starts with 'f' that contains face information.

    NOTE: face indices must be offset by 1 because OBJ files are 1-indexed.
    """

    space_splits = line.split()[1:]

    face_indices = []
    tex_coord_indices = []
    vertex_normal_indices = []

    for space_split in space_splits:
        slash_split = space_split.split("/")

        if len(slash_split) == 1:  # f v1 v2 v3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_indices.append(None)
            vertex_normal_indices.append(None)

        elif len(slash_split) == 2:  # f v1/vt1 v2/vt2 v3/vt3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_indices.append(int(slash_split[1]) - 1)
            vertex_normal_indices.append(None)

        elif len(slash_split) == 3:  # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_indices.append(int(slash_split[1]) - 1)
            vertex_normal_indices.append(int(slash_split[2]) - 1)

        else:
            raise NotImplementedError("Unsupported feature")

    return (
        face_indices,
        tex_coord_indices,
        vertex_normal_indices,
    )

@typechecked
def _parse_tex_coordinates(line: str) -> List[float]:
    return [float(x) for x in line.split()[1:]]

@typechecked
def _parse_vertex_normal(line: str) -> List[float]:
    return [float(x) for x in line.split()[1:]]

@typechecked
def _load_texture_image(
    obj_path: Path,
    vertical_flip: bool = True,
) -> Float[Tensor, "image_height image_width 3"]:
    """
    Loads the texture image associated with the given .obj file.

    Args:
        obj_path: Path to the .obj file whose texture is being loaded.
        vertical_flip: Whether to flip the texture image vertically.
            This is necessary for rendering systems following OpenGL conventions.
    """
    image_paths = list(obj_path.parent.glob("*.png"))

    texture_image = None
    if len(image_paths) > 0:
        texture_image = imageio.imread(image_paths[0])
        texture_image = texture_image.astype(np.float32) / 255.0
        if vertical_flip:
            texture_image = np.flip(texture_image, axis=0).copy()
    texture_image = torch.from_numpy(texture_image)

    return texture_image
