import striker as sr

from tqdm.auto import tqdm
import taichi as ti
import numpy as np

sr.init(backend=ti.gpu)
scene = sr.Scene(show_viewer=True)

entity = scene.add_entity(
    init_pos=(3.5, 0.1),
    init_yaw=np.pi,
    init_vel=(0.01, 0),
    radius=0.25,
)

entity2 = scene.add_entity(
    init_pos=(-3.5, 0.1), init_yaw=0, init_vel=(0.01, 0), radius=0.25, mass=0.1
)

scene.build(n_envs=4096)

for i in tqdm(range(100_000_000)):
    scene.step()
