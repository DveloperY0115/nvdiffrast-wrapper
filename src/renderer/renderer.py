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
        mesh: Mesh,
        camera: PerspectiveCamera,
    ) -> Tuple[
        Float[Tensor, "image_height image_width 3"],
        Union[
            Float[Tensor, "image_height image_width 4"],
            Float[Tensor, "image_height image_width 0"],
        ],
    ]:
        """
        Renders triangle meshes to images via differentiable rasterization.
        """

        # assertions
        assert self.device == camera.device, f"{self.device} != {camera.device}"
        assert self.device == mesh.device, f"{self.device} != {mesh.device}"
        image_height, image_width = int(camera.image_height), int(camera.image_width)

        vertices_clip = self.apply_view_projection_matrices(
            mesh.vertices[None, ...],
            camera.camera_to_world,
            camera.projection_matrix,
        )

        raster_out, image_space_derivatives = self.rasterize(
            vertices_clip,
            mesh.faces,
            resolution=(image_height, image_width),
        )

        if mesh.has_texture:
            per_pixel_tex_coordinates = self.interpolate_attributes(
                mesh.tex_coordinates,
                raster_out,
                mesh.tex_coordinate_indices,
            )
            image = self.sample_texture(
                mesh.texture_image[None, ...],
                per_pixel_tex_coordinates,
            )
        else:
            assert mesh.vertex_colors is not None
            image = self.interpolate_attributes(
                mesh.vertex_colors[None, ...],
                raster_out,
                mesh.faces,
            )

        image = self.antialias(
            image,
            raster_out,
            vertices_clip,
            mesh.faces,
        )

        # mask out background
        # TODO: allow selecting background color
        mask = raster_out[..., 3:4] > 0.0
        image = torch.where(
            mask,
            image,
            torch.zeros_like(image),
        )

        # flip images vertically
        # nvdiffrast renders images upside down
        # flipping could be done when defining projection matrices
        # but we flip the images to keep the convention tidy
        # Refer to: https://github.com/NVlabs/nvdiffrast/issues/44
        image = torch.flip(image, dims=[1])
        image_space_derivatives = torch.flip(image_space_derivatives, dims=[1])

        # truncate batch dimension
        image = image[0, ...]
        image_space_derivatives = image_space_derivatives[0, ...]

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
    def interpolate_attributes(
        self,
        attributes: Float[Tensor, "* attribute_dim"],
        rasterization_output: Float[Tensor, "batch_size image_height image_width 4"],
        faces: Int[Tensor, "num_face 3"],
    ) -> Float[Tensor, "batch_size image_height image_width attribute_dim"]:
        """
        Interpolates attributes given triangle indices.
        """

        result, _ = dr.interpolate(
            attributes,
            rasterization_output,
            faces,
        )

        return result

    @jaxtyped
    @typechecked
    def sample_texture(
        self,
        texture_image: Float[Tensor, "batch_size texture_height texture_width 3"],
        tex_coordinates: Float[Tensor, "batch_size image_height image_width 2"],
    ) -> Float[Tensor, "batch_size image_height image_width 3"]:
        """
        Samples texture image given per-pixel texture coordinates.
        """

        image = dr.texture(
            texture_image,
            tex_coordinates,
            filter_mode="linear",
        )
        return image

    @jaxtyped
    @typechecked
    def antialias(
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
