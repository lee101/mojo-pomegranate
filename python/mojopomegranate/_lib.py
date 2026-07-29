"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "kernels.mojo")
LIB = os.environ.get("MOJOPOMEGRANATE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pomegranate.so"
)

I = ctypes.c_int64

_SIGNATURES = {
    "mp_normal_emissions": ([I] * 9, None),
    "mp_categorical_emissions": ([I] * 7, None),
    "mp_mixture_posteriors": ([I] * 5, None),
    "mp_mixture_probabilities": ([I] * 5, None),
    "mp_weighted_stats": ([I] * 8, None),
    "mp_weighted_categorical_stats": ([I] * 7, None),
    "mp_hmm_forward": ([I] * 7, None),
    "mp_hmm_backward": ([I] * 7, None),
    "mp_hmm_finish": ([I] * 12, None),
    "mp_hmm_viterbi": ([I] * 10, None),
    "mp_bn_log_probability": ([I] * 9, None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJOPOMEGRANATE_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    command = ["bash", script]
    if shutil.which("mojo") is None:
        pixi = shutil.which("pixi")
        if pixi is None:
            raise BuildError("neither mojo nor pixi is available")
        command = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_lib, name)
            function.argtypes = argtypes
            function.restype = restype
    return _lib


def f64(value, *, copy: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind == "c":
        raise TypeError("complex values are not supported")
    if copy:
        return np.array(array, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(array, dtype=np.float64)


def i64(value) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise TypeError("categorical data must be a real numeric or boolean array")
    if array.size:
        if array.dtype.kind == "f":
            if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
                raise ValueError("categorical data must contain finite integer values")
        info = np.iinfo(np.int64)
        if np.any(array < info.min) or np.any(array > info.max):
            raise OverflowError("categorical data is outside the int64 range")
    return np.ascontiguousarray(array, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise TypeError("native buffers must be C-contiguous NumPy arrays")
    address = int(value.ctypes.data)
    if value.size and address == 0:
        raise ValueError("native buffers must have a non-null address")
    return address
