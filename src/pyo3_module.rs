// PyO3 module exposing the Rust VBGMM to Python.
//
// Replaces c-concoct/vbgmm.pyx (Cython wrapper around c_vbgmm_fit).
// The Python-facing interface matches the Cython original:
//
//     vbgmm.fit(xarray, nClusters, seed, threads, piter=500)
//
// Returns a 1-D numpy array of int32 cluster assignments.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::input;
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

/// Load composition from a FASTA file: parse sequences, count k-mers,
/// normalize per-contig, and log-transform.
///
/// Returns (data, contig_ids, contig_lengths) where:
/// - data: 2-D numpy array, shape (n_contigs, n_features), float64
/// - contig_ids: list of contig ID strings
/// - contig_lengths: list of contig lengths (as floats, matching pandas Series dtype)
///
/// Ports: concoct/input.py load_composition (L65-78)
#[pyfunction]
#[pyo3(signature = (comp_file, kmer_len, length_threshold))]
fn load_composition<'py>(
    py: Python<'py>,
    comp_file: &str,
    kmer_len: usize,
    length_threshold: usize,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>, Vec<f64>)> {
    let file = std::fs::File::open(comp_file)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{comp_file}: {e}")))?;
    let reader = std::io::BufReader::new(file);
    let result = input::load_composition(reader, kmer_len, length_threshold);

    let array = PyArray2::from_vec2(
        py,
        &result.data.chunks(result.n_features).map(|row| row.to_vec()).collect::<Vec<_>>(),
    )?;
    let lengths_f64: Vec<f64> = result.contig_lengths.iter().map(|&l| l as f64).collect();

    Ok((array, result.contig_ids, lengths_f64))
}

/// Python module: `vbgmm`
///
/// Drop-in replacement for the Cython vbgmm module.
#[pymodule]
#[pyo3(name = "vbgmm")]
fn pyo3_vbgmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(load_composition, m)?)?;
    Ok(())
}
