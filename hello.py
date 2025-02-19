import striker as sr

from tqdm.auto import tqdm
import taichi as ti
import numpy as np

sr.init(backend=ti.gpu)
scene = sr.Scene(show_viewer=False)

entity = scene.add_entity(
    init_pos=(3.5, 0.1),
    init_yaw=np.random.random() * 2 * np.pi,
    init_vel=(np.random.random() * 0.1, 0),
    radius=0.25,
)

scene.build(n_envs=4096)

for i in tqdm(range(100_000_000)):
    scene.step()
