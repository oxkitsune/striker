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

    def _init_entities(self):
        struct_entity_info = ti.types.struct(
            pos=ti.types.vector(2, sr.ti_float),
            yaw=sr.ti_float,
            vel=ti.types.vector(2, sr.ti_float),
            radius=sr.ti_float,
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
            )
            self._kernel_init_entities_state()

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
        """
        We store whether two entities (i, j) are possibly colliding in a given environment env_idx.
        1 = overlapping AABBs, 0 = non-overlapping AABBs
        """
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
        self._func_update_aabbs()

    @ti.func
    def _func_apply_velocity(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                absolute_delta = ti.Vector([0.0, 0.0])
                entity_yaw = self.entities_state[i, env_idx].yaw
                entity_vel = self.entities_state[i, env_idx].vel

                # rotate velocity vector by yaw
                absolute_delta[0] = entity_vel[0] * ti.cos(entity_yaw) - entity_vel[
                    1
                ] * ti.sin(entity_yaw)
                absolute_delta[1] = entity_vel[0] * ti.sin(entity_yaw) + entity_vel[
                    1
                ] * ti.cos(entity_yaw)

                # new position = old position + velocity
                radius = self.entities_info[i].radius
                new_pos = self.entities_state[i, env_idx].pos + absolute_delta
                collided_with_boundary = False
                normal = ti.Vector([0.0, 0.0])

                # Check for collisions and calculate the normal
                if new_pos[0] + radius >= 150:  # Right boundary
                    normal = ti.Vector([-1.0, 0.0])  # Normal points left
                    new_pos[0] = 150 - radius
                    collided_with_boundary = True

                if new_pos[0] - radius <= -150:  # Left boundary
                    normal = ti.Vector([1.0, 0.0])  # Normal points right
                    new_pos[0] = -150 + radius
                    collided_with_boundary = True

                if new_pos[1] + radius >= 150:  # Top boundary
                    normal = ti.Vector([0.0, -1.0])  # Normal points down
                    new_pos[1] = 150 - radius
                    collided_with_boundary = True

                if new_pos[1] - radius <= -150:  # Bottom boundary
                    normal = ti.Vector([0.0, 1.0])  # Normal points up
                    new_pos[1] = -150 + radius
                    collided_with_boundary = True

                # Reflect velocity if a collision occurred
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

                    # Update yaw based on the reflected velocity
                    entity_yaw = ti.atan2(reflected_vel[1], reflected_vel[0])

                self.entities_state[i, env_idx].pos = new_pos
                self.entities_state[i, env_idx].yaw = entity_yaw

    @ti.func
    def _func_update_aabbs(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                # Current position
                pos_x = self.entities_state[i, env_idx].pos[0]
                pos_y = self.entities_state[i, env_idx].pos[1]

                # Radius is stored in entities_info
                r = self.entities_info[i].radius

                # Compute bounds
                min_x = pos_x - r
                min_y = pos_y - r
                max_x = pos_x + r
                max_y = pos_y + r

                # Update the AABB in entities_state
                self.entities_state[i, env_idx].aabb_min = ti.Vector([min_x, min_y])
                self.entities_state[i, env_idx].aabb_max = ti.Vector([max_x, max_y])

    @ti.func
    def _func_broad_phase_collisions(self):
        """
        Naive broad phase: For each environment, check every pair (i, j)
        of entities to see if their AABBs overlap.
        broad_phase_collisions[i, j, env_idx] = 1 if overlapping, else 0.
        """
        # Reset collisions
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(self.n_entities):
                    self.broad_phase_collisions[i, j, env_idx] = 0

        # Check overlaps
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    min_i = self.entities_state[i, env_idx].aabb_min
                    max_i = self.entities_state[i, env_idx].aabb_max
                    min_j = self.entities_state[j, env_idx].aabb_min
                    max_j = self.entities_state[j, env_idx].aabb_max

                    # AABB overlap check:
                    #   overlap iff intervals overlap on both x and y
                    #
                    # i.e., i.max_x >= j.min_x and i.min_x <= j.max_x
                    #       and i.max_y >= j.min_y and i.min_y <= j.max_y
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
        """
        Naive narrow phase: For each environment, check every pair (i, j)
        of entities to see if they are colliding, only if their AABBs overlap.
        """
        # Reset collisions
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

                        # Compare squared distances
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
                for j in range(self.n_entities):
                    if self.narrow_phase_collisions[i, j, env_idx] > 0:
                        self.entities_state[i, env_idx].yaw == np.pi

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B

        if shape is None:
            return (B,)
        elif type(shape) in [list, tuple]:
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)
