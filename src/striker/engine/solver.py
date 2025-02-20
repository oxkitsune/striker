import striker as sr
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
            self._init_broad_phase()
            self._init_narrow_phase()
            self.impulse_accum = ti.Vector.field(
                2, dtype=sr.ti_float, shape=self._batch_shape(self.n_entities)
            )

    def _init_entities(self):
        struct_entity_info = ti.types.struct(
            pos=ti.types.vector(2, sr.ti_float),
            yaw=sr.ti_float,
            vel=ti.types.vector(2, sr.ti_float),
            radius=sr.ti_float,
            mass=sr.ti_float,
            restitution=sr.ti_float,
        )
        struct_entity_state = ti.types.struct(
            pos=ti.types.vector(2, sr.ti_float),
            yaw=sr.ti_float,
            vel=ti.types.vector(2, sr.ti_float),
            aabb_min=ti.types.vector(2, sr.ti_float),
            aabb_max=ti.types.vector(2, sr.ti_float),
        )
        self.entities_info = struct_entity_info.field(
            shape=self.n_entities, needs_grad=False, layout=ti.Layout.SOA
        )
        self.entities_init_AABB = ti.Vector.field(
            2, dtype=sr.ti_float, shape=(self.n_entities, 4)
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
                    [entity.init_vel for entity in self.scene._entities],
                    dtype=np.float32,
                ),
                entities_radius=np.array(
                    [entity.radius for entity in self.scene._entities], dtype=np.float32
                ),
                entities_mass=np.array(
                    [entity.mass for entity in self.scene._entities], dtype=np.float32
                ),
                entities_restitution=np.array(
                    [entity.restitution for entity in self.scene._entities],
                    dtype=np.float32,
                ),
            )
            self._kernel_init_entities_state()

    @ti.kernel
    def _kernel_init_entities(
        self,
        entities_pos: ti.types.ndarray(),
        entities_yaw: ti.types.ndarray(),
        entities_vel: ti.types.ndarray(),
        entities_radius: ti.types.ndarray(),
        entities_mass: ti.types.ndarray(),
        entities_restitution: ti.types.ndarray(),
    ):
        for i in range(self.n_entities):
            for j in ti.static(range(2)):
                self.entities_info[i].pos[j] = entities_pos[i, j]
                self.entities_info[i].vel[j] = entities_vel[i, j]
            self.entities_info[i].yaw = entities_yaw[i]
            self.entities_info[i].radius = entities_radius[i]
            self.entities_info[i].mass = entities_mass[i]
            self.entities_info[i].restitution = entities_restitution[i]
            min_x = entities_pos[i, 0] - entities_radius[i]
            min_y = entities_pos[i, 1] - entities_radius[i]
            max_x = entities_pos[i, 0] + entities_radius[i]
            max_y = entities_pos[i, 1] + entities_radius[i]
            self.entities_init_AABB[i, 0] = ti.Vector([min_x, min_y], dt=sr.ti_float)
            self.entities_init_AABB[i, 1] = ti.Vector([max_x, min_y], dt=sr.ti_float)
            self.entities_init_AABB[i, 2] = ti.Vector([min_x, max_y], dt=sr.ti_float)
            self.entities_init_AABB[i, 3] = ti.Vector([max_x, max_y], dt=sr.ti_float)

    @ti.kernel
    def _kernel_init_entities_state(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in ti.static(range(2)):
                    self.entities_state[i, env_idx].pos[j] = self.entities_info[i].pos[
                        j
                    ]
                    self.entities_state[i, env_idx].vel[j] = self.entities_info[i].vel[
                        j
                    ]
                self.entities_state[i, env_idx].yaw = self.entities_info[i].yaw

    def _init_broad_phase(self):
        self.broad_phase_collisions = ti.field(
            dtype=ti.i32, shape=self._batch_shape((self.n_entities, self.n_entities))
        )

    def _init_narrow_phase(self):
        self.narrow_phase_collisions = ti.field(
            dtype=ti.i32, shape=self._batch_shape((self.n_entities, self.n_entities))
        )

    def step(self):
        self._kernel_step()

    @ti.kernel
    def _kernel_step(self):
        self._func_broad_phase_collisions()
        self._func_narrow_phase_collisions()
        self._func_update_velocity()
        self._func_apply_velocity()
        self._func_correct_penetration()
        self._func_update_aabbs()

    @ti.func
    def _func_apply_velocity(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                absolute_delta = ti.Vector([0.0, 0.0])
                entity_yaw = self.entities_state[i, env_idx].yaw
                entity_vel = self.entities_state[i, env_idx].vel
                absolute_delta[0] = entity_vel[0] * ti.cos(entity_yaw) - entity_vel[
                    1
                ] * ti.sin(entity_yaw)
                absolute_delta[1] = entity_vel[0] * ti.sin(entity_yaw) + entity_vel[
                    1
                ] * ti.cos(entity_yaw)
                radius = self.entities_info[i].radius
                new_pos = self.entities_state[i, env_idx].pos + absolute_delta
                collided_with_boundary = False
                normal = ti.Vector([0.0, 0.0])
                if new_pos[0] + radius >= 150:
                    normal = ti.Vector([-1.0, 0.0])
                    new_pos[0] = 150 - radius
                    collided_with_boundary = True
                if new_pos[0] - radius <= -150:
                    normal = ti.Vector([1.0, 0.0])
                    new_pos[0] = -150 + radius
                    collided_with_boundary = True
                if new_pos[1] + radius >= 150:
                    normal = ti.Vector([0.0, -1.0])
                    new_pos[1] = 150 - radius
                    collided_with_boundary = True
                if new_pos[1] - radius <= -150:
                    normal = ti.Vector([0.0, 1.0])
                    new_pos[1] = -150 + radius
                    collided_with_boundary = True
                if collided_with_boundary:
                    global_vel = ti.Vector(
                        [
                            entity_vel[0] * ti.cos(entity_yaw)
                            - entity_vel[1] * ti.sin(entity_yaw),
                            entity_vel[0] * ti.sin(entity_yaw)
                            + entity_vel[1] * ti.cos(entity_yaw),
                        ]
                    )
                    reflected_vel = global_vel - 2 * (global_vel.dot(normal)) * normal
                    entity_yaw = ti.atan2(reflected_vel[1], reflected_vel[0])
                self.entities_state[i, env_idx].pos = new_pos
                self.entities_state[i, env_idx].yaw = entity_yaw

    @ti.func
    def _func_update_aabbs(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                pos_x = self.entities_state[i, env_idx].pos[0]
                pos_y = self.entities_state[i, env_idx].pos[1]
                r = self.entities_info[i].radius
                min_x = pos_x - r
                min_y = pos_y - r
                max_x = pos_x + r
                max_y = pos_y + r
                self.entities_state[i, env_idx].aabb_min = ti.Vector([min_x, min_y])
                self.entities_state[i, env_idx].aabb_max = ti.Vector([max_x, max_y])

    @ti.func
    def _func_broad_phase_collisions(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(self.n_entities):
                    self.broad_phase_collisions[i, j, env_idx] = 0
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    min_i = self.entities_state[i, env_idx].aabb_min
                    max_i = self.entities_state[i, env_idx].aabb_max
                    min_j = self.entities_state[j, env_idx].aabb_min
                    max_j = self.entities_state[j, env_idx].aabb_max
                    overlap = (
                        (max_i[0] >= min_j[0])
                        and (min_i[0] <= max_j[0])
                        and (max_i[1] >= min_j[1])
                        and (min_i[1] <= max_j[1])
                    )
                    if overlap:
                        self.broad_phase_collisions[i, j, env_idx] = 1
                        self.broad_phase_collisions[j, i, env_idx] = 1

    @ti.func
    def _func_narrow_phase_collisions(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(self.n_entities):
                    self.narrow_phase_collisions[i, j, env_idx] = 0
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    if self.broad_phase_collisions[i, j, env_idx] == 1:
                        pos_i = self.entities_state[i, env_idx].pos
                        pos_j = self.entities_state[j, env_idx].pos
                        r_i = self.entities_info[i].radius
                        r_j = self.entities_info[j].radius
                        delta = pos_i - pos_j
                        dist_sqr = delta.dot(delta)
                        r_sum = r_i + r_j
                        if dist_sqr < r_sum * r_sum:
                            self.narrow_phase_collisions[i, j, env_idx] = 1
                            self.narrow_phase_collisions[j, i, env_idx] = 1

    @ti.func
    def _func_update_velocity(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                self.impulse_accum[i, env_idx] = ti.Vector([0.0, 0.0])
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    if self.narrow_phase_collisions[i, j, env_idx] > 0:
                        pos_i = self.entities_state[i, env_idx].pos
                        pos_j = self.entities_state[j, env_idx].pos
                        n = pos_j - pos_i
                        d = n.norm()
                        n_normal = n / d if d > 1e-6 else ti.Vector([1.0, 0.0])
                        v_i = ti.Vector(
                            [
                                self.entities_state[i, env_idx].vel[0]
                                * ti.cos(self.entities_state[i, env_idx].yaw)
                                - self.entities_state[i, env_idx].vel[1]
                                * ti.sin(self.entities_state[i, env_idx].yaw),
                                self.entities_state[i, env_idx].vel[0]
                                * ti.sin(self.entities_state[i, env_idx].yaw)
                                + self.entities_state[i, env_idx].vel[1]
                                * ti.cos(self.entities_state[i, env_idx].yaw),
                            ]
                        )
                        v_j = ti.Vector(
                            [
                                self.entities_state[j, env_idx].vel[0]
                                * ti.cos(self.entities_state[j, env_idx].yaw)
                                - self.entities_state[j, env_idx].vel[1]
                                * ti.sin(self.entities_state[j, env_idx].yaw),
                                self.entities_state[j, env_idx].vel[0]
                                * ti.sin(self.entities_state[j, env_idx].yaw)
                                + self.entities_state[j, env_idx].vel[1]
                                * ti.cos(self.entities_state[j, env_idx].yaw),
                            ]
                        )
                        relative_velocity = v_i - v_j
                        if relative_velocity.dot(n_normal) > 0:
                            continue
                        e = ti.min(
                            self.entities_info[i].restitution,
                            self.entities_info[j].restitution,
                        )
                        m_i = self.entities_info[i].mass
                        m_j = self.entities_info[j].mass
                        impulse_magnitude = (
                            -(1 + e)
                            * relative_velocity.dot(n_normal)
                            / (1 / m_i + 1 / m_j)
                        )
                        impulse = impulse_magnitude * n_normal
                        self.impulse_accum[i, env_idx] += impulse / m_i
                        self.impulse_accum[j, env_idx] -= impulse / m_j
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                v = ti.Vector(
                    [
                        self.entities_state[i, env_idx].vel[0]
                        * ti.cos(self.entities_state[i, env_idx].yaw)
                        - self.entities_state[i, env_idx].vel[1]
                        * ti.sin(self.entities_state[i, env_idx].yaw),
                        self.entities_state[i, env_idx].vel[0]
                        * ti.sin(self.entities_state[i, env_idx].yaw)
                        + self.entities_state[i, env_idx].vel[1]
                        * ti.cos(self.entities_state[i, env_idx].yaw),
                    ]
                )
                v_new = v + self.impulse_accum[i, env_idx]
                new_speed = v_new.norm()
                new_yaw = ti.atan2(v_new[1], v_new[0])
                self.entities_state[i, env_idx].yaw = new_yaw
                self.entities_state[i, env_idx].vel = ti.Vector([new_speed, 0.0])

    @ti.func
    def _func_correct_penetration(self):
        correction_factor = 1

        # avoid jitter
        slop = 0.01
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    if self.narrow_phase_collisions[i, j, env_idx] > 0:
                        pos_i = self.entities_state[i, env_idx].pos
                        pos_j = self.entities_state[j, env_idx].pos
                        r_i = self.entities_info[i].radius
                        r_j = self.entities_info[j].radius
                        n = pos_j - pos_i
                        d = n.norm()
                        penetration = (r_i + r_j) - d
                        if penetration > slop:
                            # Normalize the collision normal, or use a default if too small.
                            n_normal = n / d if d > 1e-6 else ti.Vector([1.0, 0.0])
                            m_i = self.entities_info[i].mass
                            m_j = self.entities_info[j].mass
                            inv_mass_sum = 1 / m_i + 1 / m_j
                            correction = (
                                correction_factor
                                * (penetration - slop)
                                / inv_mass_sum
                                * n_normal
                            )
                            self.entities_state[i, env_idx].pos -= correction / m_i
                            self.entities_state[j, env_idx].pos += correction / m_j

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B
        if shape is None:
            return (B,)
        elif type(shape) in [list, tuple]:
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)
