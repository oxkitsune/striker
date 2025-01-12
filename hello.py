import striker as sr

from tqdm.auto import tqdm
import taichi as ti

ti.init(arch=ti.gpu)

scene = sr.Scene(show_viewer=True)
entity = scene.add_entity(init_pos=(0.5, 0.5), init_yaw=0, radius=0.1)

scene.build(n_envs=1024)

for i in tqdm(range(100_000_000)):
    scene.step()
