"""
perspective_camera.py

The perspective camera class.
"""

from typing import Union

from jaxtyping import Float, Int, jaxtyped
import numpy as np
import torch
from torch import Tensor
from typeguard import typechecked

from src.renderer.cameras.base_camera import Camera


class PerspectiveCamera(Camera):
    """The perspective camera class."""

    camera_to_world: Float[Tensor, "4 4"]
    """The camera-to-world transformation matrix"""
    projection_matrix: Float[Tensor, "4 4"]
    """The OpenGL projection matrix of the camera"""
    aspect_ratio: Union[float, Float[Tensor, "1"]]
    """The aspect ratio of the camera"""
    field_of_view: Union[float, Float[Tensor, "1"]]
    """The field of view of the camera in degrees"""
    near: Union[float, Float[Tensor, "1"]]
    """The near bound of rays casted from the camera"""
    far: Union[float, Float[Tensor, "1"]]
    """The far bound of rays casted from the camera"""
    image_width: Union[int, Int[Tensor, "1"]]
    """The width of the image rendered with the camera"""
    image_height: Union[int, Int[Tensor, "1"]]
    """The height of the image rendered with the camera"""
    device: Union[str, torch.device]
    """The index of GPU where the camera resides"""

    @jaxtyped
    @typechecked
    def __init__(
        self,
        camera_to_world: Float[Tensor, "4 4"],
        aspect_ratio: Union[float, Float[Tensor, "1"]],
        field_of_view: Union[float, Float[Tensor, "1"]],
        near: Union[float, Float[Tensor, "1"]],
        far: Union[float, Float[Tensor, "1"]],
        image_width: Union[int, Int[Tensor, "1"]],
        image_height: Union[int, Int[Tensor, "1"]],
        device: Union[str, torch.device],
    ) -> None:

        self.camera_to_world = camera_to_world
        self.aspect_ratio = aspect_ratio
        self.field_of_view = field_of_view
        self.near = near
        self.far = far
        self.image_width = image_width
        self.image_height = image_height
        self.device = device

        # cast numeric values to tensors
        for variable, value in vars(self).items():
            if isinstance(value, int):
                setattr(self, variable, torch.tensor(getattr(self, variable), dtype=torch.int))
            elif isinstance(value, float):
                setattr(self, variable, torch.tensor(getattr(self, variable), dtype=torch.float))

        # transfer tensors to device
        for variable, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(
                    self,
                    variable,
                    getattr(self, variable).to(self.device),
                )

        # build projection matrix
        self._build_projection_matrix()

    @jaxtyped
    @typechecked
    def _build_projection_matrix(self) -> None:
        """
        Builds the OpenGL projection matrix for the current camera.
        """

        aspect_ratio: Float[Tensor, "1"] = self.aspect_ratio
        field_of_view: Float[Tensor, "1"] = self.field_of_view * np.pi / 180.0
        near: Float[Tensor, "1"] = self.near
        far: Float[Tensor, "1"] = self.far

        # fill in the elements of projection matrix
        projection_matrix = torch.zeros(4, 4, device=self.device)
        projection_matrix[0, 0] = 1.0 / (aspect_ratio * torch.tan(field_of_view / 2.0))
        projection_matrix[1, 1] = 1.0 / torch.tan(field_of_view / 2.0)
        projection_matrix[2, 2] = - (near + far) / (far - near)
        projection_matrix[2, 3] = - (2.0 * near * far) / (far - near)
        projection_matrix[3, 2] = -1.0

        self.projection_matrix = projection_matrix
