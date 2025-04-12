"""Simple example use of striker sim."""

import numpy as np
from tqdm.auto import tqdm

import striker as sr

sr.init(backend=sr.gpu)

scene = sr.Scene(show_viewer=True)

robot = scene.add_entity(
    init_pos=(3.5, 0.1), init_yaw=np.pi, init_vel=(0.0015, 0), radius=0.25, mass=1, color=(0.2, 0.1, 0.2)
)

obstacle = scene.add_entity(
    init_pos=(-3.5, 0.1),
    init_yaw=0,
    init_vel=(0.015, 0),
    radius=0.25,
    mass=10,
    restitution=0.9,
)

scene.build(n_envs=2)

yaw = np.pi

for _i in tqdm(range(100_000_000)):
    scene.step()

    yaw += 0.005
    robot.set_yaw([[yaw], [yaw]])
