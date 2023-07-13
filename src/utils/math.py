"""
math.py

A collection of math utilities.
"""


from jaxtyping import Float, Shaped, jaxtyped
import torch
from torch import Tensor
from typeguard import typechecked


@jaxtyped
@typechecked
def homogenize_coordinates(
    coordinates: Shaped[Tensor, "*batch_size 3"],
) -> Shaped[Tensor, "*batch_size 4"]:
    """
    Appends a row to the bottom of coordinates.
    """
    ones = torch.ones(
        coordinates.shape[:-1] + (1,),
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    coordinates = torch.cat([coordinates, ones], dim=-1)

    return coordinates

@jaxtyped
@typechecked
def homogenize_matrices(
    matrices: Shaped[Tensor, "*batch_size 3 4"],
) -> Shaped[Tensor, "*batch_size 4 4"]:
    """
    Appends a row to the bottom of matrices.
    """
    ones = torch.ones(
        matrices.shape[:-2] + (1, 4),
        dtype=matrices.dtype,
        device=matrices.device,
    )
    matrices = torch.cat([matrices, ones], dim=-2)

    return matrices

@jaxtyped
@typechecked
def compute_inverse_affine(
    matrix: Float[Tensor, "4 4"],
) -> Float[Tensor, "4 4"]:
    """
    Computes the inverse of the given Affine transformation matrix.
    """

    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]

    inverse = torch.zeros_like(matrix)
    inverse[3, 3] = 1.0
    inverse[:3, :3] = rotation.t()
    inverse[:3, 3] = -rotation.t() @ translation

    return inverse
