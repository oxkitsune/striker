import numpy as np
import taichi as ti
import torch

import striker as sr


@ti.data_oriented
class PhysicsSolver:
    def __init__(self, scene):
        self.scene = scene
        self._entities = []

    def build(self):
        self.num_envs = self.scene.n_envs
        self._envs_idx = self.scene._envs_idx
        self._B = max(1, self.num_envs)
        self.n_entities = len(self.scene._entities)
        if self.n_entities > 0:
            self._init_entities()
            self.broad_phase_collisions = ti.field(
                dtype=ti.i32,
                shape=self._batch_shape((self.n_entities, self.n_entities), first_dim=True),
            )

            self.narrow_phase_collisions = ti.field(
                dtype=ti.i32,
                shape=self._batch_shape((self.n_entities, self.n_entities), first_dim=True),
            )
            self.impulse_accum = ti.Vector.field(
                2,
                dtype=sr.ti_float,
                shape=self._batch_shape(self.n_entities, first_dim=True),
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
        self.entities_info = struct_entity_info.field(shape=self.n_entities, needs_grad=False, layout=ti.Layout.SOA)
        self.entities_init_AABB = ti.Vector.field(2, dtype=sr.ti_float, shape=(self.n_entities, 4))
        self.entities_state = struct_entity_state.field(
            shape=self._batch_shape(self.n_entities, first_dim=True),
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
                entities_radius=np.array([entity.radius for entity in self.scene._entities], dtype=np.float32),
                entities_mass=np.array([entity.mass for entity in self.scene._entities], dtype=np.float32),
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
                self.entities_state[env_idx, i].pos = self.entities_info[i].pos
                self.entities_state[env_idx, i].vel = self.entities_info[i].vel
                self.entities_state[env_idx, i].yaw = self.entities_info[i].yaw

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
                entity_yaw = self.entities_state[env_idx, i].yaw
                entity_vel = self.entities_state[env_idx, i].vel
                absolute_delta[0] = entity_vel[0] * ti.cos(entity_yaw) - entity_vel[1] * ti.sin(entity_yaw)
                absolute_delta[1] = entity_vel[0] * ti.sin(entity_yaw) + entity_vel[1] * ti.cos(entity_yaw)
                new_pos = self.entities_state[env_idx, i].pos + absolute_delta
                self.entities_state[env_idx, i].pos = new_pos
                self.entities_state[env_idx, i].yaw = entity_yaw

    @ti.func
    def _func_update_aabbs(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                pos_x = self.entities_state[env_idx, i].pos[0]
                pos_y = self.entities_state[env_idx, i].pos[1]
                r = self.entities_info[i].radius
                min_x = pos_x - r
                min_y = pos_y - r
                max_x = pos_x + r
                max_y = pos_y + r
                self.entities_state[env_idx, i].aabb_min = ti.Vector([min_x, min_y])
                self.entities_state[env_idx, i].aabb_max = ti.Vector([max_x, max_y])

    @ti.func
    def _func_broad_phase_collisions(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(self.n_entities):
                    self.broad_phase_collisions[env_idx, i, j] = 0
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    min_i = self.entities_state[env_idx, i].aabb_min
                    max_i = self.entities_state[env_idx, i].aabb_max
                    min_j = self.entities_state[env_idx, j].aabb_min
                    max_j = self.entities_state[env_idx, j].aabb_max
                    overlap = (
                        (max_i[0] >= min_j[0])
                        and (min_i[0] <= max_j[0])
                        and (max_i[1] >= min_j[1])
                        and (min_i[1] <= max_j[1])
                    )
                    if overlap:
                        self.broad_phase_collisions[env_idx, i, j] = 1
                        self.broad_phase_collisions[env_idx, j, i] = 1

    @ti.func
    def _func_narrow_phase_collisions(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(self.n_entities):
                    self.narrow_phase_collisions[env_idx, i, j] = 0

        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    if self.broad_phase_collisions[env_idx, i, j] == 1:
                        pos_i = self.entities_state[env_idx, i].pos
                        pos_j = self.entities_state[env_idx, j].pos
                        r_i = self.entities_info[i].radius
                        r_j = self.entities_info[j].radius
                        delta = pos_i - pos_j
                        dist_sqr = delta.dot(delta)
                        r_sum = r_i + r_j
                        if dist_sqr < r_sum * r_sum:
                            self.narrow_phase_collisions[env_idx, i, j] = 1
                            self.narrow_phase_collisions[env_idx, j, i] = 1

    @ti.func
    def _func_update_velocity(self):
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                self.impulse_accum[env_idx, i] = ti.Vector([0.0, 0.0])
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    if self.narrow_phase_collisions[env_idx, i, j] > 0:
                        pos_i = self.entities_state[env_idx, i].pos
                        pos_j = self.entities_state[env_idx, j].pos
                        n = pos_j - pos_i
                        d = n.norm()
                        n_normal = n / d if d > 1e-6 else ti.Vector([1.0, 0.0])
                        v_i = ti.Vector([
                            self.entities_state[env_idx, i].vel[0] * ti.cos(self.entities_state[env_idx, i].yaw)
                            - self.entities_state[env_idx, i].vel[1] * ti.sin(self.entities_state[env_idx, i].yaw),
                            self.entities_state[env_idx, i].vel[0] * ti.sin(self.entities_state[env_idx, i].yaw)
                            + self.entities_state[env_idx, i].vel[1] * ti.cos(self.entities_state[env_idx, i].yaw),
                        ])
                        v_j = ti.Vector([
                            self.entities_state[env_idx, j].vel[0] * ti.cos(self.entities_state[env_idx, j].yaw)
                            - self.entities_state[env_idx, j].vel[1] * ti.sin(self.entities_state[env_idx, j].yaw),
                            self.entities_state[env_idx, j].vel[0] * ti.sin(self.entities_state[env_idx, j].yaw)
                            + self.entities_state[env_idx, j].vel[1] * ti.cos(self.entities_state[env_idx, j].yaw),
                        ])
                        relative_velocity = v_i - v_j
                        if relative_velocity.dot(n_normal) > 0:
                            continue
                        e = ti.min(
                            self.entities_info[i].restitution,
                            self.entities_info[j].restitution,
                        )
                        m_i = self.entities_info[i].mass
                        m_j = self.entities_info[j].mass
                        impulse_magnitude = -(1 + e) * relative_velocity.dot(n_normal) / (1 / m_i + 1 / m_j)
                        impulse = impulse_magnitude * n_normal
                        self.impulse_accum[env_idx, i] += impulse / m_i
                        self.impulse_accum[env_idx, j] -= impulse / m_j

        for env_idx in range(self._B):
            for i in range(self.n_entities):
                v = ti.Vector([
                    self.entities_state[env_idx, i].vel[0] * ti.cos(self.entities_state[env_idx, i].yaw)
                    - self.entities_state[env_idx, i].vel[1] * ti.sin(self.entities_state[env_idx, i].yaw),
                    self.entities_state[env_idx, i].vel[0] * ti.sin(self.entities_state[env_idx, i].yaw)
                    + self.entities_state[env_idx, i].vel[1] * ti.cos(self.entities_state[env_idx, i].yaw),
                ])
                v_new = v + self.impulse_accum[env_idx, i]
                new_speed = v_new.norm()
                self.entities_state[env_idx, i].vel = ti.Vector([new_speed, 0.0])

    @ti.func
    def _func_correct_penetration(self):
        correction_factor = 1.5

        # avoid jitter
        slop = 0.001
        for env_idx in range(self._B):
            for i in range(self.n_entities):
                for j in range(i + 1, self.n_entities):
                    if self.narrow_phase_collisions[env_idx, i, j] > 0:
                        pos_i = self.entities_state[env_idx, i].pos
                        pos_j = self.entities_state[env_idx, j].pos
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
                            correction = correction_factor * (penetration - slop) / inv_mass_sum * n_normal

                            self.entities_state[env_idx, i].pos -= correction / m_i * 1.1
                            self.entities_state[env_idx, j].pos += correction / m_j * 1.1

    def get_entities_pos(self, entities_idx, envs_idx=None):
        tensor, entities_idx, envs_idx = self._sanitize_2D_io_variables(
            None, entities_idx, self.n_entities, 2, envs_idx, idx_name="entities_idx"
        )
        self._kernel_get_entities_pos(tensor, entities_idx, envs_idx)

        if self.num_envs == 0:
            tensor = tensor.squeeze(0)

        return tensor

    @ti.kernel
    def _kernel_get_entities_pos(
        self,
        tensor: ti.types.ndarray(),
        entities_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        for e_idx, b_idx in ti.ndrange(entities_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(2)):
                tensor[b_idx, e_idx, i] = self.entities_state[envs_idx[b_idx], entities_idx[e_idx]].pos[i]

    def get_entities_yaw(self, entities_idx, envs_idx=None):
        tensor, entities_idx, envs_idx = self._sanitize_1D_io_variables(
            None, entities_idx, self.n_entities, envs_idx, idx_name="entities_idx"
        )
        self._kernel_get_entities_yaw(tensor, entities_idx, envs_idx)

        if self.num_envs == 0:
            tensor = tensor.squeeze(0)

        return tensor

    @ti.kernel
    def _kernel_get_entities_yaw(
        self,
        tensor: ti.types.ndarray(),
        entities_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        for e_idx, b_idx in ti.ndrange(entities_idx.shape[0], envs_idx.shape[0]):
            tensor[b_idx, e_idx] = self.entities_state[envs_idx[b_idx], entities_idx[e_idx]].yaw

    def get_entities_vel(self, entities_idx, envs_idx=None):
        tensor, entities_idx, envs_idx = self._sanitize_2D_io_variables(
            None, entities_idx, self.n_entities, 2, envs_idx, idx_name="entities_idx"
        )
        self._kernel_get_entities_vel(tensor, entities_idx, envs_idx)

        if self.num_envs == 0:
            tensor = tensor.squeeze(0)

        return tensor

    @ti.kernel
    def _kernel_get_entities_vel(
        self,
        tensor: ti.types.ndarray(),
        entities_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        for e_idx, b_idx in ti.ndrange(entities_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(2)):
                tensor[b_idx, e_idx, i] = self.entities_state[envs_idx[b_idx], entities_idx[e_idx]].vel[i]

    def set_entities_pos(self, entities_idx, pos, envs_idx=None):
        pos, entities_idx, envs_idx = self._sanitize_2D_io_variables(
            pos, entities_idx, self.n_entities, 2, envs_idx, idx_name="entities_idx"
        )
        self._kernel_set_entities_pos(pos, entities_idx, envs_idx)

    @ti.kernel
    def _kernel_set_entities_pos(
        self,
        tensor: ti.types.ndarray(),
        entities_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        for e_idx, b_idx in ti.ndrange(entities_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(2)):
                self.entities_state[envs_idx[b_idx], entities_idx[e_idx]].pos[i] = tensor[b_idx, e_idx, i]

    def set_entities_yaw(self, entities_idx, yaw, envs_idx=None):
        yaw, entities_idx, envs_idx = self._sanitize_1D_io_variables(
            yaw, entities_idx, self.n_entities, envs_idx, idx_name="entities_idx"
        )
        self._kernel_set_entities_yaw(yaw, entities_idx, envs_idx)

    @ti.kernel
    def _kernel_set_entities_yaw(
        self,
        tensor: ti.types.ndarray(),
        entities_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        for e_idx, b_idx in ti.ndrange(entities_idx.shape[0], envs_idx.shape[0]):
            self.entities_state[envs_idx[b_idx], entities_idx[e_idx]].yaw = tensor[b_idx, e_idx]

    def set_entities_vel(self, entities_idx, vel, envs_idx=None):
        vel, entities_idx, envs_idx = self._sanitize_2D_io_variables(
            vel, entities_idx, self.n_entities, 2, envs_idx, idx_name="entities_idx"
        )
        self._kernel_set_entities_vel(vel, entities_idx, envs_idx)

    @ti.kernel
    def _kernel_set_entities_vel(
        self,
        tensor: ti.types.ndarray(),
        entities_idx: ti.types.ndarray(),
        envs_idx: ti.types.ndarray(),
    ):
        for e_idx, b_idx in ti.ndrange(entities_idx.shape[0], envs_idx.shape[0]):
            for i in ti.static(range(2)):
                self.entities_state[envs_idx[b_idx], entities_idx[e_idx]].vel[i] = tensor[b_idx, e_idx, i]

    def _sanitize_envs_idx(self, envs_idx=None, *, unsafe=False):
        if envs_idx is None:
            envs_idx = self._envs_idx
        else:
            if self.num_envs == 0:
                raise ValueError("`envs_idx` is not supported for non-parallized scenes!")

            envs_idx = torch.as_tensor(envs_idx, dtype=torch.int32, device=sr.device).contiguous()

            if not unsafe and envs_idx.ndim != 1:
                raise ValueError("Expected a 1D tensor for `envs_idx`!")

            if not unsafe and ((envs_idx < 0).any() or (envs_idx >= self.num_envs).any()):
                raise ValueError("`envs_idx` exceeds valid range.")

        return envs_idx

    def _sanitize_1D_io_variables(
        self,
        tensor,
        inputs_idx,
        input_max,
        envs_idx=None,
        batched=True,
        idx_name="entities_idx",
        *,
        skip_allocation=False,
        unsafe=False,
    ):
        # Handling default arguments
        if batched:
            envs_idx = self._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        else:
            envs_idx = torch.empty((0,), dtype=torch.int32, device=sr.device)

        if inputs_idx is None:
            inputs_idx = range(input_max)
        elif isinstance(inputs_idx, slice):
            inputs_idx = range(
                inputs_idx.start or 0,
                inputs_idx.stop if inputs_idx.stop is not None else input_max,
                inputs_idx.step or 1,
            )
        elif isinstance(inputs_idx, int):
            inputs_idx = [inputs_idx]

        is_preallocated = tensor is not None
        if not is_preallocated and not skip_allocation:
            if batched and self.num_envs > 0:
                shape = self._batch_shape(len(inputs_idx), True, B=len(envs_idx))
            else:
                shape = (len(inputs_idx),)
            tensor = torch.empty(shape, dtype=torch.float32, device=sr.device)

        # Early return if unsafe
        if unsafe:
            return tensor, inputs_idx, envs_idx

        # Perform a bunch of sanity checks
        _inputs_idx = torch.atleast_1d(torch.as_tensor(inputs_idx, dtype=torch.int32, device=sr.device)).contiguous()

        if _inputs_idx.ndim != 1:
            raise ValueError(f"Expecting 1D tensor for `{idx_name}`.")
        inputs_start, inputs_end = min(inputs_idx), max(inputs_idx)
        if inputs_start < 0 or input_max <= inputs_end:
            raise ValueError(f"`{idx_name}` is out-of-range.")

        if is_preallocated:
            _tensor = torch.as_tensor(tensor, dtype=torch.float32, device=sr.device).contiguous()
            tensor = _tensor
            if tensor.shape[-1] != len(_inputs_idx):
                raise ValueError(f"Last dimension of the input tensor does not match length of `{idx_name}`.")

            if batched:
                if self.num_envs == 0:
                    if tensor.ndim != 1:
                        raise ValueError(
                            f"Invalid input shape: {tensor.shape}. Expecting a 1D tensor for non-parallelized scene."
                        )
                else:
                    if tensor.ndim == 2:
                        if tensor.shape[0] != len(envs_idx):
                            raise ValueError(
                                f"Invalid input shape: {tensor.shape}. First dimension of the input tensor "
                                "does not match length of `envs_idx` (or `scene.n_envs` if `envs_idx` is None)."
                            )
                    else:
                        raise ValueError(
                            f"Invalid input shape: {tensor.shape}. Expecting a 2D tensor for scene "
                            "with parallelized envs."
                        )
            else:
                if tensor.ndim != 1:
                    raise ValueError("Expecting 1D output tensor.")
        return tensor, _inputs_idx, envs_idx

    def _sanitize_2D_io_variables(
        self,
        tensor,
        inputs_idx,
        input_max,
        vec_size,
        envs_idx=None,
        batched=True,
        idx_name="entities_idx",
        *,
        skip_allocation=False,
        unsafe=False,
    ):
        # Handling default arguments
        if batched:
            envs_idx = self._sanitize_envs_idx(envs_idx, unsafe=unsafe)
        else:
            envs_idx = torch.empty((0,), dtype=torch.int32, device=sr.device)

        if inputs_idx is None:
            inputs_idx = range(input_max)
        elif isinstance(inputs_idx, slice):
            inputs_idx = range(
                inputs_idx.start or 0,
                inputs_idx.stop if inputs_idx.stop is not None else input_max,
                inputs_idx.step or 1,
            )
        elif isinstance(inputs_idx, int):
            inputs_idx = [inputs_idx]

        is_preallocated = tensor is not None
        if not is_preallocated and not skip_allocation:
            if batched and self.num_envs > 0:
                shape = self._batch_shape((len(inputs_idx), vec_size), True, B=len(envs_idx))
            else:
                shape = (len(inputs_idx), vec_size)
            tensor = torch.empty(shape, dtype=torch.float32, device=sr.device)

        # Early return if unsafe
        if unsafe:
            return tensor, inputs_idx, envs_idx

        # Perform a bunch of sanity checks
        _inputs_idx = torch.as_tensor(inputs_idx, dtype=torch.int32, device=sr.device).contiguous()
        if _inputs_idx.ndim != 1:
            raise ValueError(f"Expecting 1D tensor for `{idx_name}`.")
        inputs_start, inputs_end = min(inputs_idx), max(inputs_idx)
        if inputs_start < 0 or input_max <= inputs_end:
            raise ValueError(f"`{idx_name}` is out-of-range.")

        if is_preallocated:
            _tensor = torch.as_tensor(tensor, dtype=torch.float32, device=sr.device).contiguous()
            tensor = _tensor
            if tensor.shape[-2] != len(_inputs_idx):
                raise ValueError(
                    f"Second last dimension of the input tensor ({tensor.shape[-2]}) "
                    "does not match length of `{idx_name}` ({len(_inputs_idx)})."
                )
            if tensor.shape[-1] != vec_size:
                raise ValueError(f"Last dimension of the input tensor must be {vec_size}.")

            if batched:
                if self.num_envs == 0:
                    if tensor.ndim != 2:
                        raise ValueError(
                            f"Invalid input shape: {tensor.shape}. Expecting a 2D tensor for non-parallelized scene."
                        )
                else:
                    if tensor.ndim == 3:
                        if tensor.shape[0] != len(envs_idx):
                            raise ValueError(
                                f"Invalid input shape: {tensor.shape}. First dimension of the input "
                                "tensor does not match length of `envs_idx` (or `scene.n_envs` if `envs_idx` is None)."
                            )
                    else:
                        raise ValueError(
                            f"Invalid input shape: {tensor.shape}. Expecting a 3D tensor for "
                            "scene with parallelized envs."
                        )
            elif tensor.ndim != 2:
                raise ValueError("Expecting 2D input tensor.")

        return tensor, _inputs_idx, envs_idx

    def _batch_shape(self, shape=None, first_dim=False, B=None):
        if B is None:
            B = self._B
        if shape is None:
            return (B,)
        elif type(shape) in [list, tuple]:
            return (B,) + shape if first_dim else shape + (B,)
        else:
            return (B, shape) if first_dim else (shape, B)
