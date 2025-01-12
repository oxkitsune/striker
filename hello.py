import striker as sr

from tqdm.auto import tqdm
import taichi as ti

ti.init(arch=ti.gpu)

scene = sr.Scene(show_viewer=False)
entity = scene.add_entity(
    init_pos=(0.8, 0.5), init_yaw=0, init_vel=(-0.001, 0), radius=0.05
)
entity = scene.add_entity(
    init_pos=(0.1, 0.5), init_yaw=0, init_vel=(0.001, 0), radius=0.05
)

scene.build(n_envs=1024)

for i in tqdm(range(100_000_000)):
    scene.step()
