mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn train_matches_c(
        n_samples in 8usize..32,
        n_dims in 2usize..5,
        n_clusters in 2usize..5,
        source_data in prop::collection::vec(-5.0f64..5.0, 32 * 5),
        seed in 1u64..100,
    ) {
        let nd = n_dims;
        let nn = n_samples;
        let nk = n_clusters;
        let max_iter = 1000;
        let epsilon = 1.0e-4;

        let data: Vec<f64> = source_data.iter().copied().take(nn * nd).collect();
        prop_assume!(data.len() == nn * nd);

        // Build VB params same way both sides do
        let beta0 = 0.001f64;
        let nu0 = nd as f64;
        let (var, _) = vbgmm::calc_sample_var(&data, nn, nd);
        let mut inv_w0 = vec![0.0f64; nd * nd];
        for i in 0..nd { inv_w0[i * nd + i] = var[i] * (nd as f64); }
        let log_wishart_b = vbgmm::d_log_wishart_b(&inv_w0, nd, nu0, true);
        let vb_params = vbgmm::VBParams { beta0, nu0, inv_w0 };

        // Rust
        let (mut z, mut ms) = vbgmm::init_kmeans(nn, nd, nk, &data, seed, max_iter, &vb_params);
        let rust_result = vbgmm::gmm_train_vb(
            nn, nd, nk, &data, &mut z, &mut ms,
            &vb_params, log_wishart_b, max_iter, epsilon,
        );

        // C
        let data_ptrs: Vec<*const f64> = (0..nn).map(|i| data[i * nd..].as_ptr()).collect();
        let mut c_assign = vec![0i32; nn];
        unsafe {
            c_ffi::ffi_trainFull(
                data_ptrs.as_ptr(),
                nn as i32, nk as i32, nd as i32,
                seed, max_iter as i32, epsilon,
                c_assign.as_mut_ptr(),
            );
        }

        // Bit-exact assignment comparison
        for i in 0..nn {
            assert_eq!(
                rust_result.assignments[i], c_assign[i],
                "assignment mismatch at sample={i}: rust={} vs c={}",
                rust_result.assignments[i], c_assign[i],
            );
        }
    }
}
