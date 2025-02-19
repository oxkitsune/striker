import striker as sr
import taichi as ti


@ti.data_oriented
class Visualizer:
    def __init__(self, scene, viewport_size: tuple[float, float]):
        self.scene = scene
        self.viewport_size = viewport_size

    def build(self):
        self._window = ti.ui.Window(name="Striker", res=(640, 640), vsync=True)
        self.render_pos = ti.Vector.field(
            2, dtype=sr.ti_float, shape=self.scene.solver.n_entities
        )
        self.render_radii = ti.field(
            dtype=sr.ti_float, shape=self.scene.solver.n_entities
        )
        self.render_colors = ti.Vector.field(
            3, dtype=sr.ti_float, shape=self.scene.solver.n_entities
        )
        self.render_aabb_vertices = ti.Vector.field(
            2, dtype=sr.ti_float, shape=self.scene.solver.n_entities * 8
        )
        self.render_yaw_overlay = ti.Vector.field(
            2, dtype=sr.ti_float, shape=self.scene.solver.n_entities * 2
        )
        self._kernel_setup_colors()

    @ti.func
    def _func_world_to_screen(self, position: ti.math.vec2) -> ti.math.vec2:  # type: ignore
        return (
            (position[0] + self.viewport_size[0] / 2) / self.viewport_size[0],
            (position[1] + self.viewport_size[1] / 2) / self.viewport_size[1],
        )

    @ti.kernel
    def _kernel_setup_colors(self):
        for i in range(self.scene.solver.n_entities):
            self.render_colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])

    @ti.kernel
    def _kernel_prepare_render(self):
        for i in range(self.scene.solver.n_entities):
            self.render_pos[i] = self._func_world_to_screen(
                self.scene.solver.entities_state[i, 0].pos
            )
            self.render_radii[i] = self.scene.solver.entities_info[i].radius / min(
                self.viewport_size[0], self.viewport_size[1]
            )
            aabb_min = self._func_world_to_screen(
                self.scene.solver.entities_state[i, 0].aabb_min
            )
            aabb_max = self._func_world_to_screen(
                self.scene.solver.entities_state[i, 0].aabb_max
            )

            self.render_aabb_vertices[i * 8 + 0] = aabb_min
            self.render_aabb_vertices[i * 8 + 1] = ti.Vector([aabb_max[0], aabb_min[1]])
            self.render_aabb_vertices[i * 8 + 2] = aabb_min
            self.render_aabb_vertices[i * 8 + 3] = ti.Vector([aabb_min[0], aabb_max[1]])
            self.render_aabb_vertices[i * 8 + 4] = aabb_max
            self.render_aabb_vertices[i * 8 + 5] = ti.Vector([aabb_min[0], aabb_max[1]])
            self.render_aabb_vertices[i * 8 + 6] = aabb_max
            self.render_aabb_vertices[i * 8 + 7] = ti.Vector([aabb_max[0], aabb_min[1]])

            self.render_yaw_overlay[i * 2 + 0] = self.render_pos[i]

            radius = self.scene.solver.entities_info[i].radius
            self.render_yaw_overlay[i * 2 + 1] = self._func_world_to_screen(
                self.scene.solver.entities_state[i, 0].pos
                + ti.Vector(
                    [
                        radius * ti.cos(self.scene.solver.entities_state[i, 0].yaw),
                        radius * ti.sin(self.scene.solver.entities_state[i, 0].yaw),
                    ]
                )
            )

            for j in range(self.scene.solver.n_entities):
                if self.scene.solver.narrow_phase_collisions[i, j, 0] > 0:
                    self.render_colors[i] = ti.Vector([1, 0, 0])
                else:
                    self.render_colors[i] = ti.Vector([0, 1, 0])

    def render(self):
        color = (0.0705882353, 0.6274509804, 0)
        canvas = self._window.get_canvas()
        canvas.set_background_color(color)

        self._kernel_prepare_render()
        canvas.circles(
            centers=self.render_pos,
            per_vertex_radius=self.render_radii,
            per_vertex_color=self.render_colors,
            radius=0.5,
        )
        canvas.lines(vertices=self.render_yaw_overlay, color=(0, 0, 0), width=0.001)
        canvas.lines(
            vertices=self.render_aabb_vertices,
            color=(1, 1, 1),
            width=0.001,
        )

        self._window.show()
