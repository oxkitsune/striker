# striker ⚽️

Massively parallel 2D physics simulation, built for reinforcement learning.

## Example

```py
import striker as sr

from tqdm.auto import tqdm
import taichi as ti
import numpy as np

sr.init(backend=ti.gpu)

scene = sr.Scene(show_viewer=False)

scene.add_entity(
    init_pos=(3.5, 0.1), init_yaw=np.pi, init_vel=(0.02, 0), radius=0.25, mass=1
)

scene.add_entity(
    init_pos=(-3.5, 0.1),
    init_yaw=0,
    init_vel=(0.01, 0),
    radius=0.25,
    mass=1.1,
    restitution=0.3,
)

scene.build(n_envs=4096)

for i in tqdm(range(100_000_000)):
    scene.step()

    print(scene.entity_states.pos)
    break

```

Inspired by [genesis](https://github.com/Genesis-Embodied-AI/Genesis).
