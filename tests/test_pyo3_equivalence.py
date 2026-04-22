"""
Equivalence test: PyO3 vbgmm.fit vs Cython vbgmm.fit.

Calls both implementations on the same input data and asserts that the
cluster assignments are identical (element-wise).
"""

import importlib
import importlib.util
import os
import sys

import numpy as np


def load_pyo3_vbgmm():
    """Load the PyO3 vbgmm module from the path given by VBGMM_PYO3_LIB."""
    lib_dir = os.environ.get("VBGMM_PYO3_LIB")
    if lib_dir is None:
        raise RuntimeError(
            "Set VBGMM_PYO3_LIB to the directory containing the PyO3 vbgmm .so"
        )
    # Find the .so file in the directory
    for f in os.listdir(lib_dir):
        if f.startswith("vbgmm") and f.endswith(".so"):
            so_path = os.path.join(lib_dir, f)
            break
    else:
        raise RuntimeError(f"No vbgmm .so found in {lib_dir}")

    # Load from explicit path to avoid sys.modules cache conflicts.
    # The module name must be "vbgmm" to match PyInit_vbgmm, but we
    # register it under a different key to avoid clobbering the Cython one.
    spec = importlib.util.spec_from_file_location("vbgmm", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_cython_vbgmm():
    """Load the Cython vbgmm module (from the installed concoct package)."""
    import vbgmm
    return vbgmm


def test_pyo3_matches_cython():
    """Both modules must produce identical assignments on the same input."""
    cython_vbgmm = load_cython_vbgmm()
    pyo3_vbgmm = load_pyo3_vbgmm()

    # Deterministic test data: 200 samples, 10 dimensions
    rng = np.random.RandomState(42)
    data = rng.randn(200, 10).astype(np.float64, order="C")

    n_clusters = 5
    seed = 42
    threads = 1
    piter = 500

    cython_assign = cython_vbgmm.fit(
        np.copy(data, order="C"), n_clusters, seed, threads, piter
    )
    pyo3_assign = pyo3_vbgmm.fit(
        np.copy(data, order="C"), n_clusters, seed, threads, piter
    )

    np.testing.assert_array_equal(
        cython_assign,
        pyo3_assign,
        err_msg="PyO3 and Cython vbgmm.fit produced different assignments",
    )


def test_pyo3_piter_default():
    """Calling fit without piter should use default (500), same as explicit."""
    pyo3_vbgmm = load_pyo3_vbgmm()

    rng = np.random.RandomState(7)
    data = rng.randn(100, 8).astype(np.float64, order="C")

    assign_explicit = pyo3_vbgmm.fit(
        np.copy(data, order="C"), 4, 7, 1, 500
    )
    assign_default = pyo3_vbgmm.fit(
        np.copy(data, order="C"), 4, 7, 1
    )

    np.testing.assert_array_equal(
        assign_explicit,
        assign_default,
        err_msg="Default piter should produce same result as piter=500",
    )


if __name__ == "__main__":
    test_pyo3_matches_cython()
    test_pyo3_piter_default()
    print("All tests passed.")
