/// FFI bindings to the C VBGMM implementation.
/// Used as the oracle for equivalence testing during the rewrite.
pub mod c_ffi;

/// Rust reimplementations of C functions.
/// Each function must produce bit-identical output to its C counterpart.
pub mod vbgmm;

/// Input loading: composition (FASTA + k-mer) and coverage (TSV).
/// Ports concoct/input.py.
pub mod input;

/// PCA via LAPACK SVD. Replaces sklearn.decomposition.PCA.
pub mod pca;

/// Output file writing. Ports concoct/output.py.
pub mod output;

/// PyO3 module exposing vbgmm.fit to Python.
/// Replaces the Cython wrapper (c-concoct/vbgmm.pyx).
#[cfg(feature = "python")]
mod pyo3_module;
