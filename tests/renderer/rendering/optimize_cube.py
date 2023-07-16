"""
optimize_cube.py

A script for reproducing the example provided by nvdiffrast:
https://github.com/NVlabs/nvdiffrast/blob/main/samples/torch/cube.py
"""

from pathlib import Path

import imageio
from jaxtyping import jaxtyped
import numpy as np
import torch
from tqdm import tqdm
from typeguard import typechecked

from src.geometry.mesh import Mesh
from src.renderer.cameras.perspective_camera import PerspectiveCamera
from src.renderer.cameras.utils import (
    compute_lookat_matrix,
    sample_location_on_sphere,
    sample_trajectory_along_upper_hemisphere,
)
from src.renderer.renderer import Renderer
from tests.renderer.rendering.utils import load_cube


@jaxtyped
@typechecked
def optimize_cube(out_dir: Path, device: torch.device) -> None:
    """A test case that loads and fits a cube via differentiable rendering"""

    radius = 2.0
    """Radius of the sphere that the camera is placed on"""
    num_step = 1000
    """Number of optimization steps"""
    resolution = 16
    """The rendering resolution"""

    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    # load cube
    data_dir = Path(__file__).parents[2] / "data/renderer/cube"
    path = data_dir / "cube_c.npz"
    faces, vertices, vertex_colors = load_cube(path)

    # create GT mesh
    mesh_gt = Mesh(
        vertices,
        faces,
        vertex_colors,
        device,
    )

    # initialize renderer
    renderer = Renderer(False, device)

    # create mesh to optimize
    vertices_optim = vertices.clone() + torch.from_numpy(
        np.random.uniform(-0.5, 0.5, size=vertices.shape)
    ).type(torch.float32)
    vertex_colors_optim = torch.rand_like(vertex_colors)
    mesh_optim = Mesh(
        vertices_optim,
        faces,
        vertex_colors_optim,
        device,
    )
    mesh_optim.vertices.requires_grad = True
    mesh_optim.vertex_colors.requires_grad = True

    # configure optimizer and scheduler
    optimizer = torch.optim.Adam(
        [mesh_optim.vertices, mesh_optim.vertex_colors],
        lr=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda x: max(0.01, 10**(-x*0.0005)),
    )

    # sample camera poses for visualization
    vis_positions = sample_trajectory_along_upper_hemisphere(
        radius=radius,
        elevation=np.pi / 4,
        num_step=num_step,
    )

    # initialize video writer
    writer = imageio.get_writer(
        vis_dir / "progress.mp4",
        format="FFMPEG",
        mode="I",
        fps=60,
        macro_block_size=1,
    )

    progress_bar = tqdm(range(num_step))
    for iteration in progress_bar:

        # sample random camera pose
        camera_position = sample_location_on_sphere(radius)
        camera_pose = compute_lookat_matrix(
            camera_position,
            torch.tensor([0.0, 0.0, 0.0]),
        )
        camera = PerspectiveCamera(
            camera_pose,
            aspect_ratio=1.0,
            field_of_view=60.0,
            near=1e-1,
            far=6.0,
            image_height=resolution,
            image_width=resolution,
            device=device,
        )

        # render GT and optimized images
        image_gt, _ = renderer.render(mesh_gt, camera)
        image_optim, _ = renderer.render(mesh_optim, camera)

        # back prop
        loss = torch.mean((image_gt - image_optim)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_description(f"Loss: {loss.item():.5f}")

        # render image
        vis_position = vis_positions[iteration]
        vis_pose = compute_lookat_matrix(
            vis_position,
            torch.tensor([0.0, 0.0, 0.0]),
        )
        vis_camera = PerspectiveCamera(
            vis_pose,
            aspect_ratio=1.0,
            field_of_view=60.0,
            near=1e-1,
            far=6.0,
            image_height=400,
            image_width=400,
            device=device,
        )
        with torch.no_grad():
            vis_image, _ = renderer.render(mesh_optim, vis_camera)

        # save video
        vis_image = torch.clamp(vis_image, 0.0, 1.0)
        vis_image = (vis_image * 255).cpu().numpy().astype(np.uint8)
        writer.append_data(vis_image)

    writer.close()
