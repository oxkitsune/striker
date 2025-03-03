import striker as sr

from tqdm.auto import tqdm
import taichi as ti
import numpy as np

sr.init(backend=sr.gpu)

scene = sr.Scene(show_viewer=True)

robot = scene.add_entity(
    init_pos=(3.5, 0.1), init_yaw=np.pi, init_vel=(0.02, 0), radius=0.25, mass=1
)

obstacle = scene.add_entity(
    init_pos=(-3.5, 0.1),
    init_yaw=0,
    init_vel=(0.0, 0),
    radius=0.25,
    mass=110,
    restitution=0.3,
)

scene.build(n_envs=2)

for i in tqdm(range(100_000_000)):
    scene.step()

    print(
        "robot pos:",
        robot.get_pos()[0],
        "robot vel:",
        robot.get_vel()[0],
        "obstacle:",
        obstacle.get_yaw(),
    )
