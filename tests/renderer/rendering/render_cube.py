"""
render_cube.py

A script for testing the renderer by rendering a simple cube.
"""

from pathlib import Path

import imageio
from jaxtyping import jaxtyped
import numpy as np
import torch
from typeguard import typechecked

from src.geometry.mesh import Mesh
from src.renderer.cameras.perspective_camera import PerspectiveCamera
from src.renderer.cameras.utils import (
    compute_lookat_matrix,
    sample_trajectory_along_upper_hemisphere
)
from src.renderer.renderer import Renderer
from tests.renderer.rendering.utils import load_cube


@jaxtyped
@typechecked
def render_cube(out_dir: Path, device: torch.device) -> None:
    """A test case that loads and renders a cube"""

    # load cube
    data_dir = Path(__file__).parents[2] / "data/renderer/cube"
    path = data_dir / "cube_c.npz"
    faces, vertices, vertex_colors = load_cube(path)
    mesh = Mesh(
        vertices[None],
        faces,
        vertex_colors[None],
        device,
    )

    # initialize renderer
    use_opengl = False
    renderer = Renderer(use_opengl, device)

    positions = sample_trajectory_along_upper_hemisphere(
        radius=2.0,
        elevation=np.pi / 4,
        num_step=1000,
    )

    # initialize video writer
    writer = imageio.get_writer(
        out_dir / "progress.mp4",
        format="FFMPEG",
        mode="I",
        fps=60,
        macro_block_size=1,
    )

    for view_index, position in enumerate(positions):

        # create camera
        aspect_ratio = 1.0
        field_of_view = 60.0
        near = 1e-1
        far = 6.0
        image_height = 400
        image_width = 400
        camera_pose = compute_lookat_matrix(
            position,
            torch.tensor([0.0, 0.0, 0.0]),
        )
        camera = PerspectiveCamera(
            camera_pose,
            aspect_ratio,
            field_of_view,
            near,
            far,
            image_height,
            image_width,
            device,
        )

        # render
        image, _ = renderer.render(mesh, camera)

        # save video
        image = torch.clamp(image.detach(), 0.0, 1.0)
        image = (image * 255.0).type(torch.uint8)
        image = image[0, ...].cpu().numpy()
        writer.append_data(image)

    writer.close()
