"""
render_armadillo.py

A script for testing the renderer by rendering Armadillo.
"""

from pathlib import Path

import imageio
from jaxtyping import jaxtyped
import numpy as np
import torch
from typeguard import typechecked

from src.renderer.cameras.perspective_camera import PerspectiveCamera
from src.renderer.cameras.utils import (
    compute_lookat_matrix,
    sample_trajectory_along_upper_hemisphere,
)
from src.renderer.renderer import Renderer
from src.utils.io.geometry.obj import load_obj


@jaxtyped
@typechecked
def render_armadillo(out_dir: Path, device: torch.device) -> None:
    """A test cases that loads and renders Armadillo"""

    # load mesh
    data_dir = Path(__file__).parents[2] / "data/renderer/armadillo"
    path = data_dir / "armadillo.obj"
    mesh = load_obj(path, device)
    mesh.vertex_colors = torch.ones_like(mesh.vertices) * 0.8

    # initialize renderer
    use_opengl = False
    renderer = Renderer(use_opengl, device)

    positions = sample_trajectory_along_upper_hemisphere(
        radius=200.0,
        elevation=np.pi / 6,
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
        far = 1000.0
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
        image = image.cpu().numpy()
        writer.append_data(image)

    writer.close()
