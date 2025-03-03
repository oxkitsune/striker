import torch

from ..visualizer import Visualizer

from ._solver import PhysicsSolver
from ._entity import Entity


class Scene:
    _entities: list[Entity]

    def __init__(
        self,
        show_viewer: bool = False,
        viewport_size: tuple[float, float] = (10.4, 7.4),
    ):
        self.show_viewer = show_viewer
        self._entities = []
        self._solver = PhysicsSolver(self)

        if show_viewer:
            self.visualizer = Visualizer(self, viewport_size=viewport_size)

    def add_entity(
        self,
        init_pos,
        init_yaw,
        init_vel,
        radius: float,
        mass: float = 1,
        restitution: float = 0.9,
    ) -> Entity:
        entity = Entity(
            scene=self,
            solver=self._solver,
            idx=self.n_entities,
            init_pos=init_pos,
            init_yaw=init_yaw,
            init_vel=init_vel,
            radius=radius,
            mass=mass,
            restitution=restitution,
        )

        self._entities.append(entity)
        return entity

    def build(self, n_envs: int = 0):
        self._parallelize(n_envs)
        self._solver.build()

        if self.show_viewer:
            self.visualizer.build()

    def _parallelize(self, n_envs: int):
        self.n_envs = n_envs

        self._B = max(1, n_envs)
        self._envs_idx = torch.arange(self._B, dtype=torch.int32)

    def step(self):
        self._solver.step()
        if self.show_viewer:
            self.visualizer.render()

    @property
    def n_entities(self):
        return len(self._entities)

    @property
    def entity_states(self):
        return self._solver.entities_state
