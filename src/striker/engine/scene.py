import torch
import taichi as ti

from .solver import PhysicsSolver
from .entity import Entity


class Scene:
    def __init__(self, show_viewer: bool = False):
        self.show_viewer = show_viewer
        self._entities = []
        self.solver = PhysicsSolver(self)

    def add_entity(self, init_pos, init_yaw, radius: float) -> Entity:
        self._entities.append(
            Entity(self, init_pos=init_pos, init_yaw=init_yaw, radius=radius)
        )

    def build(self, n_envs: int = 0):
        self._parallelize(n_envs)
        self.solver.build()

    def _parallelize(self, n_envs: int):
        self.n_envs = n_envs

        self._B = max(1, n_envs)
        self._envs_idx = torch.arange(self._B, dtype=torch.int32)

    def step(self):
        self.solver.step()
