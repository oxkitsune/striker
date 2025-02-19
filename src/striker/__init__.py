from typing import Literal
from .engine.scene import Scene
import taichi as ti

_initialized = False
device = None


def init(backend=ti.gpu, precision: Literal["32", "64"] = "32"):
    global _initialized
    if _initialized:
        raise RuntimeError("Striker already initialized.")
    _initialized = True

    global device
    device = backend

    # initialize taichi
    ti.init(arch=backend)

    global ti_float
    if precision == "32":
        ti_float = ti.f32
    elif precision == "64":
        ti_float = ti.f64
    else:
        raise ValueError(f"Unsupported precision: {precision}!")


__all__ = ["Scene", "visualizer"]
