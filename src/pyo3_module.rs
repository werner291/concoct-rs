// PyO3 module exposing the Rust VBGMM to Python.
//
// Replaces c-concoct/vbgmm.pyx (Cython wrapper around c_vbgmm_fit).
// The Python-facing interface matches the Cython original:
//
//     vbgmm.fit(xarray, nClusters, seed, threads, piter=500)
//
// Returns a 1-D numpy array of int32 cluster assignments.

use numpy::ndarray::ShapeBuilder;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::vbgmm;

/// Default number of VB iterations.
/// Matches the Python CLI default (concoct/parser.py:68, --iterations default=500).
/// Note: the C code's own default is DEF_MAX_ITER=1000 (c_vbgmm_fit.h:119),
/// but the Python layer always overrides it.
const DEFAULT_PITER: usize = 500;

/// Fit a Variational Bayesian Gaussian Mixture Model.
///
/// Parameters
/// ----------
/// xarray : numpy.ndarray[float64, ndim=2, order='C']
///     Data matrix, shape (n_samples, n_dims).
/// n_clusters : int
///     Number of initial clusters.
/// seed : int
///     Random seed for reproducibility.
/// threads : int
///     Number of threads. Configures the Rayon thread pool.
/// piter : int, optional
///     Maximum number of VB iterations (default: 500).
///
/// Returns
/// -------
/// numpy.ndarray[int32]
///     Cluster assignment for each sample.
#[pyfunction]
#[pyo3(signature = (xarray, n_clusters, seed, threads, piter=None))]
fn fit<'py>(
    py: Python<'py>,
    xarray: PyReadonlyArray2<'py, f64>,
    n_clusters: usize,
    seed: u64,
    threads: usize,
    piter: Option<usize>,
) -> Bound<'py, PyArray1<i32>> {
    let max_iter = piter.unwrap_or(DEFAULT_PITER);
    let array = xarray.as_array();
    let n_samples = array.nrows();
    let n_dims = array.ncols();

    // Get a contiguous slice — the Cython wrapper required C-contiguous input,
    // so callers already pass order='C' arrays.
    let data = array
        .as_slice_memory_order()
        .expect("xarray must be C-contiguous");

    // Configure Rayon thread pool to match the requested thread count.
    // Build a scoped pool so we don't pollute the global pool.
    let assignments = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("failed to build Rayon thread pool")
        .install(|| vbgmm::vbgmm_fit(data, n_samples, n_dims, n_clusters, seed, max_iter));

    assignments.into_pyarray(py)
}

/// Python module: `vbgmm`
///
/// Drop-in replacement for the Cython vbgmm module.
#[pymodule]
#[pyo3(name = "vbgmm")]
fn pyo3_vbgmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    Ok(())
}
