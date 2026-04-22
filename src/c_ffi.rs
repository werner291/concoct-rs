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

    /// Recompute cluster centroids from assignments.
    /// c-concoct/ffi_wrappers.c (wraps c_vbgmm_fit.c:1259-1298)
    pub fn ffi_updateMeans(
        aadX: *const *const f64,
        nN: i32, nK: i32, nD: i32,
        anMaxZ: *const i32, anW: *const i32,
        aadMu: *const *mut f64,
    );
}

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
