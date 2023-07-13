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
