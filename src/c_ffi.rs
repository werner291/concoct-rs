//! FFI declarations for C VBGMM functions.
//!
//! These are used as oracles in proptest equivalence tests.
//! Each declaration must match the signature in c-concoct/c_vbgmm_fit.h.

extern "C" {
    /// Euclidean distance between two vectors of dimension nD.
    /// c-concoct/c_vbgmm_fit.c:1300-1311
    pub fn calcDist(adX: *const f64, adMu: *const f64, nD: i32) -> f64;
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
