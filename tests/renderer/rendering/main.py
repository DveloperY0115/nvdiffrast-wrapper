"""
main.py

A script for running unit tests for renderer.
"""

from dataclasses import dataclass, field
from typing import Type

from tests.base_test import TestConfig, Test
from tests.renderer.rendering.optimize_cube import optimize_cube
from tests.renderer.rendering.render_armadillo import render_armadillo
from tests.renderer.rendering.render_bunny import render_bunny
from tests.renderer.rendering.render_cube import render_cube
from tests.renderer.rendering.render_lucy import render_lucy
from tests.renderer.rendering.render_spot import render_spot
from tests.renderer.rendering.render_teapot import render_teapot

@dataclass
class RendererTestConfig(TestConfig):
    """The configuration of a differentiable rendering test"""

    _target: Type = field(default_factory=lambda: RendererTest)

    test_name: str = "renderer/rendering"
    """The name of the test"""


class RendererTest(Test):
    """The renderer test class"""

    config: RendererTestConfig


def main():
    """The entry point of the script"""
    test_config = RendererTestConfig(
        test_cases=[
            render_armadillo,
            render_bunny,
            render_cube,
            render_lucy,
            render_spot,
            render_teapot,
            optimize_cube,
        ],
    )
    test = test_config.setup()
    test.run()


if __name__ == "__main__":
    main()
