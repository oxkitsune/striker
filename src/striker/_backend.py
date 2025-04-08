import os
import platform
import subprocess
from enum import IntEnum

import taichi as ti


class backend(IntEnum):
    cpu = 0
    gpu = 1
    cuda = 2
    metal = 3
    vulkan = 4

    def __format__(self, format_spec: str) -> str:
        return f"sr.{self.name}"

    @property
    def taichi(self):
        match self:
            case backend.cpu:
                return ti.cpu
            case backend.gpu:
                return ti.gpu
            case backend.metal:
                return ti.metal
            case backend.vulkan:
                return ti.metal


def _get_cpu_name():
    name = platform.platform().lower()

    if "macos" in name or "darwin" in name:
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        return process.stdout.strip()
    elif "linux" in name:
        command = "cat /proc/cpuinfo"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        all_info = process.stdout.strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return line.replace("\t", "").replace("model name: ", "")
    else:
        return platform.processor()
