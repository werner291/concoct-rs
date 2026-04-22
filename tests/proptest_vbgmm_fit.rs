mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn vbgmm_fit_matches_c(
        n_samples in 16usize..64,
        n_dims in 2usize..6,
        n_clusters in 2usize..6,
        source_data in prop::collection::vec(-5.0f64..5.0, 64 * 6),
        seed in 1u32..100,
    ) {
        let nd = n_dims;
        let nn = n_samples;
        let nk = n_clusters;

        let mut data: Vec<f64> = source_data.iter().copied().take(nn * nd).collect();
        prop_assume!(data.len() == nn * nd);

        // Rust
        let rust_assign = vbgmm::vbgmm_fit(&data, nn, nd, nk, seed as u64, 1000);

        // C — c_vbgmm_fit modifies data in place (generateInputData copies it)
        // so we pass a copy
        let mut c_data = data.clone();
        let mut c_assign = vec![0i32; nn];
        unsafe {
            c_ffi::c_vbgmm_fit(
                c_data.as_mut_ptr(),
                nn as i32, nd as i32, nk as i32,
                seed as i32, c_assign.as_mut_ptr(),
                1, // single thread for determinism
                1000,
            );
        }

        for i in 0..nn {
            assert_eq!(
                rust_assign[i], c_assign[i],
                "assignment mismatch at sample={i}: rust={} vs c={}",
                rust_assign[i], c_assign[i],
            );
        }
    }
}
