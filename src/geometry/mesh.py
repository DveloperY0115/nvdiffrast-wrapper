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

    vertices: Float[Tensor, "1 num_vertex 3"] = None
    """The vertices of the mesh"""
    faces: Int[Tensor, "num_face 3"] = None
    """The faces of the mesh"""
    # TODO: Extend the class to support range mode in nvdiffrast
    # TODO: Extend the class to support various attributes
    vertex_colors: Float[Tensor, "1 num_vertex 3"] = None
    """The vertex colors of the mesh"""
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
