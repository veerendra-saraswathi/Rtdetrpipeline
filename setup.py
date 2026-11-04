"""Builds the C++ preprocessing module `ird_preproc` for Python (training side).

    pip install pybind11
    pip install ./            # from the repo root (Linux: gcc; Windows: MSVC Build Tools)

The Jetson runtime builds the same source with CMake (cpp/CMakeLists.txt).
"""
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

extra = ["/O2", "/fp:precise"] if sys.platform == "win32" else ["-O2", "-ffp-contract=off"]

setup(
    name="ird_preproc",
    version="0.1.0",
    ext_modules=[
        Pybind11Extension(
            "ird_preproc",
            ["cpp/python/bind_preproc.cpp", "cpp/src/preproc.cpp"],
            include_dirs=["cpp/include"],
            cxx_std=17,
            extra_compile_args=extra,
        )
    ],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
