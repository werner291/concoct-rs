/// FFI bindings to the C VBGMM implementation.
/// Used as the oracle for equivalence testing during the rewrite.
pub mod c_ffi;

/// Rust reimplementations of C functions.
/// Each function must produce bit-identical output to its C counterpart.
pub mod vbgmm;
