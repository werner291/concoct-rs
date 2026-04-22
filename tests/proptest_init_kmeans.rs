mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn init_kmeans_matches_c(
        n_samples in 8usize..32,
        n_dims in 2usize..6,
        n_clusters in 2usize..6,
        source_data in prop::collection::vec(-5.0f64..5.0, 32 * 6),
        seed in 1u64..1000,
    ) {
        let nd = n_dims;
        let nn = n_samples;
        let nk = n_clusters;
        let max_iter = 100;

        let data: Vec<f64> = source_data.iter().copied().take(nn * nd).collect();
        prop_assume!(data.len() == nn * nd);

        // Build VB params the same way the C driver does
        let beta0 = 0.001f64;
        let nu0 = nd as f64;
        let (var, _mu) = vbgmm::calc_sample_var(&data, nn, nd);
        let mut inv_w0 = vec![0.0f64; nd * nd];
        for i in 0..nd {
            inv_w0[i * nd + i] = var[i] * (nd as f64);
        }
        let log_wishart_b = vbgmm::d_log_wishart_b(&inv_w0, nd, nu0, true);
        let _ = log_wishart_b; // used by C internally

        let vb_params = vbgmm::VBParams { beta0, nu0, inv_w0: inv_w0.clone() };

        // Rust
        let (rust_z, rust_mstep) = vbgmm::init_kmeans(
            nn, nd, nk, &data, seed, max_iter, &vb_params,
        );

        // C
        let data_ptrs: Vec<*const f64> = (0..nn).map(|i| data[i * nd..].as_ptr()).collect();
        let mut c_z_flat = vec![0.0f64; nn * nk];
        let c_z_ptrs: Vec<*mut f64> = (0..nn)
            .map(|i| c_z_flat[i * nk..].as_mut_ptr())
            .collect();
        let mut c_max_z = vec![0i32; nn];
        let mut c_pi = vec![0.0f64; nk];

        unsafe {
            c_ffi::ffi_initKMeans(
                data_ptrs.as_ptr(),
                nn as i32, nk as i32, nd as i32,
                seed, max_iter as i32,
                beta0, nu0,
                c_z_ptrs.as_ptr(),
                c_max_z.as_mut_ptr(),
                c_pi.as_mut_ptr(),
            );
        }

        // Compare Z matrix
        for i in 0..nn {
            for k in 0..nk {
                let idx = i * nk + k;
                assert_eq!(
                    rust_z[idx].to_bits(), c_z_flat[idx].to_bits(),
                    "z mismatch at [{i},{k}]: rust={:e} vs c={:e}",
                    rust_z[idx], c_z_flat[idx],
                );
            }
        }

        // Compare pi
        for k in 0..nk {
            assert_eq!(
                rust_mstep.pi[k].to_bits(), c_pi[k].to_bits(),
                "pi mismatch at k={k}: rust={:e} vs c={:e}",
                rust_mstep.pi[k], c_pi[k],
            );
        }
    }
}
