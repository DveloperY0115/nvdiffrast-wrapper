"""
mesh.py
"""

from dataclasses import dataclass

from jaxtyping import Float, Int, jaxtyped
import torch
from torch import Tensor
from typeguard import typechecked


@dataclass
class Mesh:
    """A mesh"""

    vertices: Float[Tensor, "num_vertex 3"] = None
    """The vertices of the mesh"""
    faces: Int[Tensor, "num_face 3"] = None
    """The faces of the mesh"""
    vertex_colors: Float[Tensor, "num_vertex 3"] = None
    """The vertex colors of the mesh"""
    tex_coordinates: Float[Tensor, "num_vertex 2"] = None
    """The texture coordinates of the mesh"""
    vertex_normals: Float[Tensor, "num_vertex 3"] = None
    """The vertex normals of the mesh"""
    tex_coordinate_indices: Int[Tensor, "num_face 3"] = None
    """The texture indices of the mesh"""
    vertex_normal_indices: Int[Tensor, "num_face 3"] = None
    """The vertex normal indices of the mesh"""
    texture_image: Float[Tensor, "texture_height texture_width 3"] = None
    """The texture image of the mesh"""
    has_texture: bool = False
    """A flag that indicates whether the mesh has a texture"""
    has_normal: bool = False
    """A flag that indicates whether the mesh has vertex normals"""
    device: torch.device = torch.device("cuda")
    """The device where the mesh resides"""

    @jaxtyped
    @typechecked
    def __post_init__(self) -> None:

        # transfer tensors to device
        for variable, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(
                    self,
                    variable,
                    getattr(self, variable).to(self.device),
                )
