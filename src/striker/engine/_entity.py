import taichi as ti

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

    def get_pos(self, envs_idx=None):
        return self._solver.get_entities_pos([self.idx], envs_idx).squeeze(-2)

    def get_yaw(self, envs_idx=None):
        return self._solver.get_entities_yaw([self.idx], envs_idx).squeeze(-2)
