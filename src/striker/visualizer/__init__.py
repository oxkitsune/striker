import taichi as ti


@ti.data_oriented
class Visualizer:
    def __init__(self, scene):
        self.scene = scene

    def build(self):
        self._window = ti.ui.Window(name="Striker", res=(640, 360), vsync=True)
        self.render_pos = ti.Vector.field(
            2, dtype=ti.f32, shape=self.scene.solver.n_entities
        )
        self.render_radii = ti.field(dtype=ti.f32, shape=self.scene.solver.n_entities)
        self.render_colors = ti.Vector.field(
            3, dtype=ti.f32, shape=self.scene.solver.n_entities
        )
        self._kernel_setup_colors()

    @ti.kernel
    def _kernel_setup_colors(self):
        for i in range(self.scene.solver.n_entities):
            self.render_colors[i] = ti.Vector([ti.random(), ti.random(), ti.random()])

    @ti.kernel
    def _kernel_prepare_render(self):
        for i in range(self.scene.solver.n_entities):
            self.render_pos[i] = self.scene.solver.entities_state[i, 0].pos
            self.render_radii[i] = self.scene.solver.entities_info[i].radius
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
            centers=self.render_pos, radius=0.1, per_vertex_color=self.render_colors
        )

        self._window.show()
