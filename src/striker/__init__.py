from typing import Literal
import taichi as ti
import torch

from ._backend import _get_cpu_name, backend as sr_backend
from .engine.scene import Scene

_initialized = False
device = None

global ti_float
ti_float = ti.f32


def init(backend=sr_backend.gpu, precision: Literal["32", "64"] = "32"):
    global _initialized
    if _initialized:
        raise RuntimeError("Striker already initialized.")
    _initialized = True

    global device
    device, device_name, total_mem = _setup_device(backend)

    print(f"using backend {device} on {device_name} with {total_mem}")
    ti.init(arch=backend.taichi)

    global ti_float
    if precision == "32":
        ti_float = ti.f32
    elif precision == "64":
        ti_float = ti.f64
    else:
        raise ValueError(f"Unsupported precision: {precision}!")


def _setup_device(backend: sr_backend):
    if backend == sr_backend.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda device is not available!")

        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)

        device_name = props.name
        total_mem = props.total_memory / 1024**3
    elif backend == sr_backend.metal:
        if not torch.backends.mps.is_available():
            raise RuntimeError("metal device is not available!")

        # metal device is part of the cpu on macOS
        _, total_mem, device_name = _setup_device(sr_backend.cpu)
        device = torch.device("mps")

    elif backend == sr_backend.gpu:
        if torch.cuda.is_available():
            return _setup_device(sr_backend.cuda)
        elif torch.backends.mps.is_available():
            return _setup_device(sr_backend.metal)
        else:
            raise RuntimeError("no gpu devices available!")
    else:
        # will default to cpu device if the provided backend is unknown
        device = torch.device("cpu")
        device_name = _get_cpu_name()
        # TODO(gijsd): figure out better way to obtain total amount of memory
        # total_mem = psutil.virtual_memory().total / 1024**3
        total_mem = 0

    return device, total_mem, device_name


__all__ = ["Scene", "visualizer"]

for name, member in sr_backend.__members__.items():
    globals()[name] = member
