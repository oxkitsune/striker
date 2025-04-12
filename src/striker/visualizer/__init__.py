import taichi as ti

import striker as sr


@ti.data_oriented
class Visualizer:
    """Simple visualizer example for striker scenes."""

    def __init__(
        self,
        scene,
        viewport_size: tuple[float, float],
        window_title: str = "Striker",
        window_resolution: tuple[int, int] = (640, 640),
        vsync: bool = True,
    ):
        """
        # Simple visualizer utility for Striker scenes.

        Args:
            scene (sr.Scene): The scene to visualize.
            viewport_size (tuple[float, float]): The size of the viewport. Used for scaling the rendering.
            window_title (str, optional): Window title. Defaults to "Striker".
            window_resolution (tuple[int, int], optional): The default resolution of the window. Defaults to (640, 640).
            vsync (bool, optional): whether to enable vsync or not. Defaults to True.

        """
        self.scene = scene
        self.viewport_size = viewport_size
        self.window_title = window_title
        self.window_resolution = window_resolution
        self.vsync = vsync

    def build(self):
        self._window = ti.ui.Window(name=self.window_title, res=self.window_resolution, vsync=self.vsync)
        self.render_pos = ti.Vector.field(2, dtype=sr.ti_float, shape=self.scene.n_entities)
        self.render_radii = ti.field(dtype=sr.ti_float, shape=self.scene.n_entities)
        self.render_colors = ti.Vector.field(3, dtype=sr.ti_float, shape=self.scene.n_entities)
        self.render_aabb_vertices = ti.Vector.field(2, dtype=sr.ti_float, shape=self.scene.n_entities * 8)
        self.render_yaw_overlay = ti.Vector.field(2, dtype=sr.ti_float, shape=self.scene.n_entities * 2)
        self._kernel_setup_colors()

    @ti.func
    def _func_world_to_screen(self, position: ti.math.vec2) -> ti.math.vec2:
        return (
            (position[0] + self.viewport_size[0] / 2) / self.viewport_size[0],
            (position[1] + self.viewport_size[1] / 2) / self.viewport_size[1],
        )

    @ti.kernel
    def _kernel_setup_colors(self):
        for i in range(self.scene.n_entities):
            self.render_colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])

    @ti.kernel
    def _kernel_prepare_render(self):
        for i in range(self.scene.n_entities):
            self.render_pos[i] = self._func_world_to_screen(self.scene._solver.entities_state[0, i].pos)
            self.render_radii[i] = self.scene._solver.entities_info[i].radius / min(
                self.viewport_size[0], self.viewport_size[1]
            )
            aabb_min = self._func_world_to_screen(self.scene._solver.entities_state[0, i].aabb_min)
            aabb_max = self._func_world_to_screen(self.scene._solver.entities_state[0, i].aabb_max)

            self.render_aabb_vertices[i * 8 + 0] = aabb_min
            self.render_aabb_vertices[i * 8 + 1] = ti.Vector([aabb_max[0], aabb_min[1]])
            self.render_aabb_vertices[i * 8 + 2] = aabb_min
            self.render_aabb_vertices[i * 8 + 3] = ti.Vector([aabb_min[0], aabb_max[1]])
            self.render_aabb_vertices[i * 8 + 4] = aabb_max
            self.render_aabb_vertices[i * 8 + 5] = ti.Vector([aabb_min[0], aabb_max[1]])
            self.render_aabb_vertices[i * 8 + 6] = aabb_max
            self.render_aabb_vertices[i * 8 + 7] = ti.Vector([aabb_max[0], aabb_min[1]])

            self.render_yaw_overlay[i * 2 + 0] = self.render_pos[i]

            radius = self.scene._solver.entities_info[i].radius
            self.render_yaw_overlay[i * 2 + 1] = self._func_world_to_screen(
                self.scene._solver.entities_state[0, i].pos
                + ti.Vector([
                    radius * ti.cos(self.scene._solver.entities_state[0, i].yaw),
                    radius * ti.sin(self.scene._solver.entities_state[0, i].yaw),
                ])
            )

            for j in range(self.scene.n_entities):
                if self.scene._solver.narrow_phase_collisions[i, j, 0] > 0:
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
