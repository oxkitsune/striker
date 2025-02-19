import torch
import taichi as ti

from ..visualizer import Visualizer

from .solver import PhysicsSolver
from .entity import Entity


class Scene:
    def __init__(
        self,
        show_viewer: bool = False,
        viewport_size: tuple[float, float] = (10.4, 7.4),
    ):
        self.show_viewer = show_viewer
        self._entities = []
        self.solver = PhysicsSolver(self)

        if show_viewer:
            self.visualizer = Visualizer(self, viewport_size=viewport_size)

    def add_entity(self, init_pos, init_yaw, init_vel, radius: float) -> Entity:
        self._entities.append(
            Entity(
                self,
                init_pos=init_pos,
                init_yaw=init_yaw,
                init_vel=init_vel,
                radius=radius,
            )
        )

    def build(self, n_envs: int = 0):
        self._parallelize(n_envs)
        self.solver.build()

        if self.show_viewer:
            self.visualizer.build()

    def _parallelize(self, n_envs: int):
        self.n_envs = n_envs
        self.n_entities = len(self._entities)

        self._B = max(1, n_envs)
        self._envs_idx = torch.arange(self._B, dtype=torch.int32)

    def step(self):
        self.solver.step()
        if self.show_viewer:
            self.visualizer.render()
