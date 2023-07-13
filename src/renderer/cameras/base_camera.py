"""
base_camera.py

The base class for cameras.
"""

from abc import ABC, abstractmethod

from jaxtyping import Float, Int
from torch import Tensor


class Camera(ABC):
    """The base class of cameras."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        pass  # do nothing

    @abstractmethod
    def _build_projection_matrix(self):
        """
        Builds the OpenGL projection matrix for the current camera.
        """

    @abstractmethod
    def generate_screen_coords(self) -> Int[Tensor, "num_pixel 2"]:
        """
        Generates screen coordinates corresponding to image pixels.
        """

    @abstractmethod
    def generate_ray_directions(
        self,
        screen_coords: Int[Tensor, "num_pixel 2"],
    ) -> Float[Tensor, "num_pixel 3"]:
        """
        Computes ray directions for the current camera.
        The direction vectors are represented in the camera frame.
        """

    # @abstractmethod
    # def generate_rays(self) -> RayBundle:
    #     """
    #     Generates rays for the current camera.
    #     """
