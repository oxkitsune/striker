import numpy as np
import taichi as ti


@ti.data_oriented
class PhysicsSolver:
    def __init__(self, scene):
        self.scene = scene

    def build(self):
        self.num_envs = self.scene.n_envs
        self._B = max(1, self.num_envs)

        self.n_entities = len(self.scene._entities)

        if self.n_entities > 0:
            self._init_entities()

    def _init_entities(self):
        struct_entity_info = ti.types.struct(
            pos=ti.types.vector(2, ti.f32),
            yaw=ti.f32,
            vel=ti.types.vector(2, ti.f32),
            radius=ti.f32,
        )

        struct_entity_state = ti.types.struct(
            pos=ti.types.vector(2, ti.f32),
            yaw=ti.f32,
            vel=ti.types.vector(2, ti.f32),
            aabb_min=ti.types.vector(2, ti.f32),
            aabb_max=ti.types.vector(2, ti.f32),
        )

        self.entities_info = struct_entity_info.field(
            shape=self.n_entities, needs_grad=False, layout=ti.Layout.SOA
        )

        self.entities_init_AABB = ti.Vector.field(
            2, dtype=ti.f32, shape=(self.n_entities, 4)
        )

        self.entities_state = struct_entity_state.field(
            shape=self._batch_shape(self.n_entities),
            needs_grad=False,
            layout=ti.Layout.SOA,
        )

        if self.n_entities > 0:
            self._kernel_init_entities(
                entities_pos=np.array(
                    [entity.init_pos for entity in self.scene._entities],
                    dtype=np.float32,
                ),
                entities_yaw=np.array(
                    [entity.init_yaw for entity in self.scene._entities],
                    dtype=np.float32,
                ),
                entities_vel=np.array(
                    [[0, 0] for entity in self.scene._entities], dtype=np.float32
                ),
                entities_radius=np.array(
                    [entity.radius for entity in self.scene._entities], dtype=np.float32
                ),
            )

    @ti.kernel
    def _kernel_init_entities(
        self,
        entities_pos: ti.types.ndarray(),  # type: ignore
        entities_yaw: ti.types.ndarray(),  # type: ignore
        entities_vel: ti.types.ndarray(),  # type: ignore
        entities_radius: ti.types.ndarray(),  # type: ignore
    ):
        for i in range(self.n_entities):
            for j in ti.static(range(2)):
                self.entities_info[i].pos[j] = entities_pos[i, j]
                self.entities_info[i].vel[j] = entities_vel[i, j]

            self.entities_info[i].yaw = entities_yaw[i]
            self.entities_info[i].radius = entities_radius[i]

            # compute initial AABB
            min_x = entities_pos[i, 0] - entities_radius[i]
            min_y = entities_pos[i, 1] - entities_radius[i]
            max_x = entities_pos[i, 0] + entities_radius[i]
            max_y = entities_pos[i, 1] + entities_radius[i]

            self.entities_init_AABB[i, 0] = ti.Vector([min_x, min_y], dt=ti.f32)
            self.entities_init_AABB[i, 1] = ti.Vector([max_x, min_y], dt=ti.f32)
            self.entities_init_AABB[i, 2] = ti.Vector([min_x, max_y], dt=ti.f32)
            self.entities_init_AABB[i, 3] = ti.Vector([max_x, max_y], dt=ti.f32)

    def step(self):
        self._kernel_step()

    @ti.kernel
    def _kernel_step(self):
        self._func_apply_velocity()

    @ti.func
    def _func_apply_velocity(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                self.entities_state[i, env_idx].pos += self.entities_state[
                    i,
                    env_idx,
                ].vel

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B

        if shape is None:
            return (B,)
        elif type(shape) in [list, tuple]:
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)
