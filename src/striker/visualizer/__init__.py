import striker as sr
import taichi as ti

ti.init(arch=ti.gpu)

num_objects = 2
vertices = ti.Vector.field(2, ti.f32, shape=num_objects)

# Scale factor
scale = 10_000  # Define your scale value here

# Field dimensions
FIELD_LENGTH = 9000 / scale
FIELD_WIDTH = 6000 / scale
PENALTY_AREA_LENGTH = 1650 / scale
PENALTY_AREA_WIDTH = 4000 / scale
PENALTY_MARK_SIZE = 100 / scale
PENALTY_MARK_DISTANCE = 1300 / scale
CENTER_CIRCLE_DIAMETER = 1500 / scale
BORDER_STRIP_WIDTH = 700 / scale
GOAL_POST_SIZE = 100 / scale
GOAL_AREA_WIDTH = 2200 / scale
GOAL_AREA_LENGTH = 600 / scale
LINE_WIDTH = 50 / scale

# field boundary
FIELD_BOUNDARY_LINES = ti.Vector.field(2, ti.f32, shape=4)
FIELD_BOUNDARY_LINES[0] = [BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH]
FIELD_BOUNDARY_LINES[1] = [FIELD_LENGTH - BORDER_STRIP_WIDTH, BORDER_STRIP_WIDTH]

FIELD_BOUNDARY_LINES[2] = [BORDER_STRIP_WIDTH, FIELD_LENGTH - BORDER_STRIP_WIDTH]
FIELD_BOUNDARY_LINES[3] = [
    FIELD_LENGTH - BORDER_STRIP_WIDTH,
    FIELD_LENGTH - BORDER_STRIP_WIDTH,
]

FIELD_BOUNDARY_INDICES = ti.Vector.field(2, ti.i32, shape=4)
FIELD_BOUNDARY_INDICES[0] = [0, 1]
FIELD_BOUNDARY_INDICES[1] = [2, 3]
FIELD_BOUNDARY_INDICES[2] = [0, 2]
FIELD_BOUNDARY_INDICES[3] = [1, 3]

# Penalty area
PENALTY_AREA_LINES = ti.Vector.field(2, ti.f32, shape=4)
PENALTY_AREA_LINES[0] = [
    (FIELD_LENGTH - PENALTY_AREA_LENGTH) / 2,
    (FIELD_WIDTH - PENALTY_AREA_WIDTH) / 2,
]
PENALTY_AREA_LINES[1] = [
    (FIELD_LENGTH + PENALTY_AREA_LENGTH) / 2,
    (FIELD_WIDTH - PENALTY_AREA_WIDTH) / 2,
]
PENALTY_AREA_LINES[2] = [
    (FIELD_LENGTH - PENALTY_AREA_LENGTH) / 2,
    (FIELD_WIDTH + PENALTY_AREA_WIDTH) / 2,
]
PENALTY_AREA_LINES[3] = [
    (FIELD_LENGTH + PENALTY_AREA_LENGTH) / 2,
    (FIELD_WIDTH + PENALTY_AREA_WIDTH) / 2,
]

PENALTY_AREA_INDICES = ti.Vector.field(2, ti.i32, shape=4)
PENALTY_AREA_INDICES[0] = [0, 1]
PENALTY_AREA_INDICES[1] = [2, 3]
PENALTY_AREA_INDICES[2] = [0, 2]
PENALTY_AREA_INDICES[3] = [1, 3]

# Center circle
CENTER_CIRCLE_RADIUS = CENTER_CIRCLE_DIAMETER / 2
CENTER_CIRCLE_CENTER = ti.Vector.field(2, ti.f32, shape=1)
CENTER_CIRCLE_CENTER[0] = [0.5, 0.5]


@ti.kernel
def init():
    for i in range(num_objects):
        vertices[i] = [ti.random(), ti.random()]


@ti.kernel
def update():
    for i in range(num_objects):
        if ti.random() < 0.5:
            vertices[i] += [ti.random() * 0.01, ti.random() * 0.01]
        else:
            vertices[i] -= [ti.random() * 0.01, ti.random() * 0.01]

        if vertices[i][0] > 1:
            vertices[i][0] = 1
        if vertices[i][0] < 0:
            vertices[i][0] = 0


init()


class Visualizer:
    def __init__(self, scene: sr.Scene):
        self.scene = scene

        self._window = ti.ui.Window(name="Striker", res=(640, 360), vsync=True)
        self._canvas = self._window.get_canvas()

    def render(self):
        while self._window.running:
            update()
            color = (0.0705882353, 0.6274509804, 0)

            self._canvas.set_background_color(color)

            self._canvas.lines(
                FIELD_BOUNDARY_LINES,
                LINE_WIDTH,
                indices=FIELD_BOUNDARY_INDICES,
                color=(1, 1, 1),
            )
            self._canvas.lines(
                PENALTY_AREA_LINES,
                LINE_WIDTH,
                indices=PENALTY_AREA_INDICES,
                color=(1, 1, 1),
            )

            self._canvas.circles(
                CENTER_CIRCLE_CENTER,
                radius=CENTER_CIRCLE_RADIUS,
                color=(1, 1, 1),
            )

            self._canvas.circles(vertices, color=(0.1, 0.1, 0.1), radius=0.02)

            self._window.show()
