from .engine.scene import Scene
import taichi as ti

global device


def init(device=ti.gpu):
    ti.init(arch=ti.gpu)


__all__ = ["Scene", "visualizer"]
