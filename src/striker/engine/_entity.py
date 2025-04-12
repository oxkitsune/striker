import taichi as ti
import torch

from striker.engine._solver import PhysicsSolver


@ti.data_oriented
class Entity:
    def __init__(
        self,
        scene,
        solver: PhysicsSolver,
        idx,
        init_pos,
        init_yaw,
        init_vel,
        radius: float,
        mass: float = 1.0,
        restitution: float = 0.9,
        color: tuple[float, float, float] | None = None,
    ):
        self.scene = scene
        self._solver = solver
        self.idx = idx
        self.init_pos = init_pos
        self.init_yaw = init_yaw
        self.init_vel = init_vel
        self.radius = radius
        self.mass = mass
        self.restitution = restitution
        self.color = color

    def get_pos(self, envs_idx=None):
        return self._solver.get_entities_pos([self.idx], envs_idx)

    def set_pos(self, pos, zero_velocity=False, envs_idx=None):
        self._solver.set_entities_pos([self.idx], pos, envs_idx)

        if zero_velocity:
            self._solver.set_entities_vel([self.idx], torch.zeros_like(pos), envs_idx)

    def get_yaw(self, envs_idx=None):
        return self._solver.get_entities_yaw([self.idx], envs_idx)

    def set_yaw(self, yaw, envs_idx=None):
        self._solver.set_entities_yaw([self.idx], yaw, envs_idx)

    def get_vel(self, envs_idx=None):
        return self._solver.get_entities_vel([self.idx], envs_idx)

    def set_vel(self, vel, envs_idx=None):
        self._solver.set_entities_vel([self.idx], vel, envs_idx)

    def get_color(self):
        return self.color

    def set_color(self, color: tuple[float, float, float]):
        self.color = color
