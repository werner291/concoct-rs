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
use crate::pca;
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

/// Load coverage from a TSV file: parse, filter, pseudo-count,
/// normalize, and log-transform.
///
/// Parameters
/// ----------
/// cov_file : str
///     Path to the coverage TSV file.
/// contig_ids : list[str]
///     Contig IDs to keep (from composition loading).
/// contig_lengths : list[float]
///     Corresponding contig lengths (same order as contig_ids).
/// no_cov_normalization : bool
///     If True, skip per-sample and per-contig normalization.
/// add_total_coverage : bool
///     If True, append a total_coverage column.
/// read_length : float
///     Read length for pseudo-count computation.
///
/// Returns
/// -------
/// (data, filtered_contig_ids, column_names, cov_range_start, cov_range_end)
///     data: 2-D numpy array (n_contigs × n_columns)
///     filtered_contig_ids: contig IDs in row order (coverage file order, filtered)
///     column_names: list of column name strings
///     cov_range_start, cov_range_end: column range for downstream use
///
/// Ports: concoct/input.py load_coverage (L41-79)
#[pyfunction]
#[pyo3(signature = (cov_file, contig_ids, contig_lengths, no_cov_normalization, add_total_coverage, read_length))]
fn load_coverage_rs<'py>(
    py: Python<'py>,
    cov_file: &str,
    contig_ids: Vec<String>,
    contig_lengths: Vec<f64>,
    no_cov_normalization: bool,
    add_total_coverage: bool,
    read_length: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>, Vec<String>, String, String)> {
    let file = std::fs::File::open(cov_file)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{cov_file}: {e}")))?;
    let reader = std::io::BufReader::new(file);

    let lengths_map: std::collections::HashMap<String, f64> = contig_ids
        .into_iter()
        .zip(contig_lengths)
        .collect();

    let result = input::load_coverage(
        reader,
        &lengths_map,
        no_cov_normalization,
        add_total_coverage,
        read_length,
    );

    let n_contigs = result.contig_ids.len();
    let n_cols = result.n_columns;
    let array = if n_contigs > 0 {
        PyArray2::from_vec2(
            py,
            &result.data.chunks(n_cols).map(|row| row.to_vec()).collect::<Vec<_>>(),
        )?
    } else {
        PyArray2::from_vec2(py, &Vec::<Vec<f64>>::new())?
    };

    Ok((
        array,
        result.contig_ids,
        result.column_names,
        result.cov_range.0,
        result.cov_range.1,
    ))
}

/// Load composition and optionally coverage, join, and return as numpy arrays.
///
/// Returns (data, contig_ids, column_names) where:
/// - data: 2-D numpy array (n_contigs × n_columns)
/// - contig_ids: list of contig ID strings
/// - column_names: list of column name strings
#[pyfunction]
#[pyo3(signature = (comp_file, cov_file, kmer_len, length_threshold, no_cov_normalization, add_total_coverage, read_length))]
fn load_and_join<'py>(
    py: Python<'py>,
    comp_file: &str,
    cov_file: Option<&str>,
    kmer_len: usize,
    length_threshold: usize,
    no_cov_normalization: bool,
    add_total_coverage: bool,
    read_length: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Vec<String>, Vec<String>)> {
    let comp_reader = std::io::BufReader::new(
        std::fs::File::open(comp_file)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{comp_file}: {e}")))?,
    );

    let cov_reader = match cov_file {
        Some(path) => Some(std::io::BufReader::new(
            std::fs::File::open(path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?,
        )),
        None => None,
    };

    let result = input::load_and_join(
        comp_reader,
        cov_reader,
        kmer_len,
        length_threshold,
        no_cov_normalization,
        add_total_coverage,
        read_length,
    );

    if result.n_contigs == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No contigs pass the length threshold",
        ));
    }

    let array = PyArray2::from_vec2(
        py,
        &result
            .data
            .chunks(result.n_columns)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok((array, result.contig_ids, result.column_names))
}

/// Perform PCA on a data matrix.
///
/// Parameters
/// ----------
/// data : numpy.ndarray[float64, ndim=2, order='C']
///     Data matrix, shape (n_samples, n_features).
/// n_components : float
///     If < 1.0: minimum cumulative variance ratio to retain.
///     If >= 1.0: exact number of components.
///
/// Returns
/// -------
/// (transformed, components, n_components_selected)
///     transformed: 2-D numpy array (n_samples × n_components_selected)
///     components: 2-D numpy array (n_components_selected × n_features)
///     n_components_selected: int
#[pyfunction]
#[pyo3(signature = (data, n_components))]
fn perform_pca<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    n_components: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>, usize)> {
    let array = data.as_array();
    let n_samples = array.nrows();
    let n_features = array.ncols();

    let flat = array
        .as_slice_memory_order()
        .expect("data must be C-contiguous");

    let result = pca::pca(flat, n_samples, n_features, n_components);

    let transformed = PyArray2::from_vec2(
        py,
        &result.transformed
            .chunks(result.n_components)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;

    let components = PyArray2::from_vec2(
        py,
        &result.components
            .chunks(result.n_features)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>(),
    )?;

    Ok((transformed, components, result.n_components))
}

/// Python module: `vbgmm`
///
/// Drop-in replacement for the Cython vbgmm module.
#[pymodule]
#[pyo3(name = "vbgmm")]
fn pyo3_vbgmm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(load_composition, m)?)?;
    m.add_function(wrap_pyfunction!(load_coverage_rs, m)?)?;
    m.add_function(wrap_pyfunction!(load_and_join, m)?)?;
    m.add_function(wrap_pyfunction!(perform_pca, m)?)?;
    Ok(())
}
