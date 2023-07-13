"""
main.py

A script for running unit tests for perspective camera.
"""

from dataclasses import dataclass, field
from typing import Type

from tests.base_test import TestConfig, Test
from tests.renderer.rendering.optimize_cube import optimize_cube
from tests.renderer.rendering.render_cube import render_cube


@dataclass
class RendererTestConfig(TestConfig):
    """The configuration of a differentiable rendering test"""

    _target: Type = field(default_factory=lambda: RendererTest)


class RendererTest(Test):
    """The renderer test class"""

    config: RendererTestConfig


def main():
    """The entry point of the script"""
    test_config = RendererTestConfig(
        test_cases=[
            render_cube,
            optimize_cube,
        ],
    )
    test = test_config.setup()
    test.run()


if __name__ == "__main__":
    main()
