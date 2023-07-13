"""
renderer.py

A differentiable renderer built on top of PyTorch and nvdiffrast.
"""

from typing import Tuple, Union

from jaxtyping import Float, Int, jaxtyped
import torch
from torch import Tensor
from typeguard import typechecked
import nvdiffrast.torch as dr

from src.geometry.mesh import Mesh
from src.renderer.cameras.perspective_camera import PerspectiveCamera
import src.utils.math as math_utils


class Renderer:
    """
    A renderer class that wraps camera system and differentiable rasterizer.
    """

    use_opengl: bool
    """A flag that determines whether to use OpenGL or CUDA context for rendering"""
    device: torch.device
    """Device where the renderer resides"""
    gl_context: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext]
    """The rasterizer context"""

    @jaxtyped
    @typechecked
    def __init__(
        self,
        use_opengl: bool = False,
        device: torch.device = torch.device("cuda"),
    ) -> None:

        # register attributes
        self.use_open_gl = use_opengl
        self.device = device

        # create rasterizer context
        gl_context = dr.RasterizeCudaContext(device=self.device)
        if self.use_open_gl:
            gl_context = dr.RasterizeGLContext(device=self.device)
        self.gl_context = gl_context

    @jaxtyped
    @typechecked
    def render(
        self,
        meshes: Mesh,
        camera: PerspectiveCamera,
    ) -> Tuple[
        Float[Tensor, "batch_size image_height image_width 3"],
        Union[
            Float[Tensor, "batch_size image_height image_width 4"],
            Float[Tensor, "batch_size image_height image_width 0"],
        ],
    ]:
        """
        Renders triangle meshes to images via differentiable rasterization.
        """

        # assertions
        assert self.device == camera.device, f"{self.device} != {camera.device}"
        assert self.device == meshes.device, f"{self.device} != {meshes.device}"

        image_height, image_width = int(camera.image_height), int(camera.image_width)
        vertices = meshes.vertices
        faces = meshes.faces
        vertex_colors = meshes.vertex_colors
        projection_matrix = camera.projection_matrix
        view_matrix = camera.camera_to_world

        # transformation to clip space
        vertices_clip = self.apply_view_projection_matrices(
            vertices,
            view_matrix,
            projection_matrix,
        )

        # rasterization
        raster_out, image_space_derivatives = self.rasterize(
            vertices_clip,
            faces,
            resolution=(image_height, image_width),
        )

        # interpolation
        image = self.interpolate_vertex_attributes(
            vertex_colors,
            raster_out,
            faces,
        )

        # anti-aliasing
        image = self.anti_alias(
            image,
            raster_out,
            vertices_clip,
            faces,
        )

        # flip images vertically
        # nvdiffrast renders images upside down
        # flipping could be done when defining projection matrices
        # but we flip the images to keep the convention tidy
        # Refer to: https://github.com/NVlabs/nvdiffrast/issues/44
        image = torch.flip(image, dims=[1])
        image_space_derivatives = torch.flip(image_space_derivatives, dims=[1])

        return image, image_space_derivatives

    @jaxtyped
    @typechecked
    def apply_view_projection_matrices(
        self,
        vertices: Union[
            Float[Tensor, "batch_size num_vertices 3"],
            Float[Tensor, "batch_size num_vertices 4"],
        ],
        view_matrix: Float[Tensor, "4 4"],
        projection_matrix: Float[Tensor, "4 4"],
    ) -> Float[Tensor, "batch_size num_vertices 4"]:
        """
        Transforms the vertices from world space to clip space by
        applying the view and projection matrices.
        """
        if vertices.shape[-1] == 3:
            vertices = math_utils.homogenize_coordinates(vertices)

        vertices_clip = (
            vertices \
            @ math_utils.compute_inverse_affine(view_matrix).t() \
            @ projection_matrix.t()
        )

        return vertices_clip

    @jaxtyped
    @typechecked
    def rasterize(
        self,
        vertices_clip: Float[Tensor, "batch_size num_vertex 4"],
        faces: Int[Tensor, "num_face 3"],
        resolution: Tuple[int, int],
    ):
        """
        Rasterizes the given geometry and produces fragments.
        """

        (
            rasterization_output,
            image_space_derivatives,
        ) = dr.rasterize(
            self.gl_context,
            vertices_clip,
            faces,
            resolution=resolution,
        )

        return rasterization_output, image_space_derivatives

    @jaxtyped
    @typechecked
    def interpolate_vertex_attributes(
        self,
        vertex_attributes: Float[Tensor, "batch_size num_vertex attribute_dim"],
        rasterization_output: Float[Tensor, "batch_size image_height image_width 4"],
        faces: Int[Tensor, "num_face 3"],
    ):
        """
        Interpolates per-vertex attributes.
        """
        result, _ = dr.interpolate(
            vertex_attributes,
            rasterization_output,
            faces,
        )

        return result

    @jaxtyped
    @typechecked
    def anti_alias(
        self,
        image: Float[Tensor, "batch_size image_height image_width 3"],
        rasterization_output: Float[Tensor, "batch_size image_height image_width 4"],
        vertices_clip: Float[Tensor, "batch_size num_vertex 4"],
        faces: Int[Tensor, "num_face 3"],
    ) -> Float[Tensor, "batch_size image_height image_width 3"]:
        """
        Anti-aliases the given rasterization output.
        """

        image = dr.antialias(
            image,
            rasterization_output,
            vertices_clip,
            faces,
        )

        return image
