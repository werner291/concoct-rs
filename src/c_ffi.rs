//! FFI declarations for C VBGMM functions.
//!
//! These are used as oracles in proptest equivalence tests.
//! Each declaration must match the signature in c-concoct/c_vbgmm_fit.h.

extern "C" {
    /// Euclidean distance between two vectors of dimension nD.
    /// c-concoct/c_vbgmm_fit.c:1300-1311
    pub fn calcDist(adX: *const f64, adMu: *const f64, nD: i32) -> f64;

    /// Per-dimension sample mean and variance.
    /// c-concoct/c_vbgmm_fit.c:202-227
    pub fn calcSampleVar(ptData: *const CData, adVar: *mut f64, adMu: *mut f64);

    /// Cholesky decomposition, log-determinant, and inversion (in-place).
    /// c-concoct/c_vbgmm_fit.c:507-528
    pub fn decomposeMatrix(ptSigmaMatrix: *mut GslMatrix, nD: i32) -> f64;

    /// Log Wishart normalisation constant.
    /// c-concoct/c_vbgmm_fit.c:1207-1236
    pub fn dLogWishartB(ptInvW: *mut GslMatrix, nD: i32, dNu: f64, bInv: i32) -> f64;

    /// Expected log-determinant of a Wishart distribution.
    /// c-concoct/c_vbgmm_fit.c:1238-1257
    /// Note: modifies ptW in-place (calls decomposeMatrix on it).
    pub fn dWishartExpectLogDet(ptW: *mut GslMatrix, dNu: f64, nD: i32) -> f64;

    /// M-step for a single component.
    /// c-concoct/ffi_wrappers.c (wraps c_vbgmm_fit.c:530-709)
    #[allow(clippy::too_many_arguments)]
    pub fn ffi_mstep(
        k: i32, nN: i32, nD: i32, nK: i32,
        aadZ: *const *const f64, aadX: *const *const f64,
        dBeta0: f64, dNu0: f64, aadInvW0: *const *const f64,
        adMu: *mut f64, adM: *mut f64,
        pdPi: *mut f64, pdBeta: *mut f64, pdNu: *mut f64, pdLDet: *mut f64,
        covarOut: *mut f64, sigmaOut: *mut f64,
    );

    /// E-step: compute responsibilities.
    /// c-concoct/ffi_wrappers.c (wraps c_vbgmm_fit.c:979-1046)
    #[allow(clippy::too_many_arguments)]
    pub fn ffi_calcZ(
        aadX: *const *const f64,
        nN: i32, nK: i32, nD: i32,
        aadZ: *const *mut f64,
        aadM: *const *const f64,
        aptSigma: *const *mut GslMatrix,
        adPi: *const f64, adNu: *const f64, adLDet: *const f64, adBeta: *const f64,
    );

    /// Recompute cluster centroids from assignments.
    /// c-concoct/ffi_wrappers.c (wraps c_vbgmm_fit.c:1259-1298)
    pub fn ffi_updateMeans(
        aadX: *const *const f64,
        nN: i32, nK: i32, nD: i32,
        anMaxZ: *const i32, anW: *const i32,
        aadMu: *const *mut f64,
    );
}

// --- GSL matrix FFI ---

/// Opaque GSL matrix type. We only interact with it through GSL functions.
#[repr(C)]
pub struct GslMatrix {
    pub size1: usize,
    pub size2: usize,
    pub tda: usize,
    pub data: *mut f64,
    pub block: *mut u8, // gsl_block, opaque
    pub owner: i32,
}

extern "C" {
    pub fn gsl_matrix_alloc(n1: usize, n2: usize) -> *mut GslMatrix;
    pub fn gsl_matrix_free(m: *mut GslMatrix);
    pub fn gsl_matrix_set(m: *mut GslMatrix, i: usize, j: usize, x: f64);
    pub fn gsl_matrix_get(m: *const GslMatrix, i: usize, j: usize) -> f64;
    pub fn gsl_matrix_memcpy(dest: *mut GslMatrix, src: *const GslMatrix) -> i32;
    pub fn gsl_sf_lngamma(x: f64) -> f64;
    pub fn gsl_sf_psi(x: f64) -> f64;

    pub fn gsl_vector_alloc(n: usize) -> *mut GslVector;
    pub fn gsl_vector_free(v: *mut GslVector);
    pub fn gsl_vector_set(v: *mut GslVector, i: usize, x: f64);
    pub fn gsl_vector_get(v: *const GslVector, i: usize) -> f64;
    pub fn gsl_blas_dsymv(
        uplo: i32, alpha: f64, a: *const GslMatrix, x: *const GslVector,
        beta: f64, y: *mut GslVector,
    ) -> i32;
    pub fn gsl_blas_ddot(
        x: *const GslVector, y: *const GslVector, result: *mut f64,
    ) -> i32;
}

/// Opaque GSL vector type.
#[repr(C)]
pub struct GslVector {
    pub size: usize,
    pub stride: usize,
    pub data: *mut f64,
    pub block: *mut u8,
    pub owner: i32,
}

/// Helper to create a GSL matrix from a flat row-major slice.
pub unsafe fn gsl_matrix_from_flat(data: &[f64], n: usize) -> *mut GslMatrix {
    let m = gsl_matrix_alloc(n, n);
    for i in 0..n {
        for j in 0..n {
            gsl_matrix_set(m, i, j, data[i * n + j]);
        }
    }
    m
}

/// Helper to read a GSL matrix back into a flat row-major slice.
pub unsafe fn gsl_matrix_to_flat(m: *const GslMatrix, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = gsl_matrix_get(m, i, j);
        }
    }
    out
}

// --- CONCOCT data types ---

/// Mirror of t_Data from c_vbgmm_fit.h, used to call C functions via FFI.
#[repr(C)]
pub struct CData {
    pub nN: i32,
    pub nD: i32,
    pub aadX: *const *const f64,
}

/// Builds the row-pointer array that t_Data expects from a flat slice.
/// The row pointers point into the original slice — no allocation copying data.
pub struct RowPointers {
    ptrs: Vec<*const f64>,
}

impl RowPointers {
    pub fn new(data: &[f64], n_samples: usize, n_dims: usize) -> Self {
        let ptrs: Vec<*const f64> = (0..n_samples)
            .map(|n| data[n * n_dims..].as_ptr())
            .collect();
        Self { ptrs }
    }

    pub fn as_cdata(&self, n_samples: usize, n_dims: usize) -> CData {
        CData {
            nN: n_samples as i32,
            nD: n_dims as i32,
            aadX: self.ptrs.as_ptr(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_smoke_test() {
        // (3, 4) vs (0, 0) → distance = 5
        let x = [3.0f64, 4.0];
        let mu = [0.0f64, 0.0];
        let result = unsafe { calcDist(x.as_ptr(), mu.as_ptr(), 2) };
        assert_eq!(result.to_bits(), 5.0f64.to_bits());
    }
}
