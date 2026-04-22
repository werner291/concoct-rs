mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn update_means_matches_c(
        n_samples in 2usize..128,
        n_clusters in 2usize..16,
        n_dims in 1usize..32,
        seed_data in prop::collection::vec(common::adversarial_f64(), 128 * 32),
        seed_assign in prop::collection::vec(0u32..16, 128),
    ) {
        let data: Vec<f64> = seed_data.iter().copied().take(n_samples * n_dims).collect();
        prop_assume!(data.len() == n_samples * n_dims);

        let assignments: Vec<i32> = seed_assign.iter()
            .take(n_samples)
            .map(|&a| (a as usize % n_clusters) as i32)
            .collect();
        let mut weights = vec![0i32; n_clusters];
        for &a in &assignments { weights[a as usize] += 1; }

        let mut rust_mu = vec![0.0f64; n_clusters * n_dims];
        vbgmm::update_means(&data, n_samples, n_clusters, n_dims, &assignments, &weights, &mut rust_mu);

        let data_ptrs: Vec<*const f64> = (0..n_samples)
            .map(|n| data[n * n_dims..].as_ptr())
            .collect();
        let mut c_mu_flat = vec![0.0f64; n_clusters * n_dims];
        let c_mu_ptrs: Vec<*mut f64> = (0..n_clusters)
            .map(|k| c_mu_flat[k * n_dims..].as_mut_ptr())
            .collect();
        unsafe {
            c_ffi::ffi_updateMeans(
                data_ptrs.as_ptr(),
                n_samples as i32, n_clusters as i32, n_dims as i32,
                assignments.as_ptr(), weights.as_ptr(),
                c_mu_ptrs.as_ptr(),
            );
        }

        for k in 0..n_clusters {
            for j in 0..n_dims {
                let idx = k * n_dims + j;
                assert_eq!(
                    rust_mu[idx].to_bits(), c_mu_flat[idx].to_bits(),
                    "mismatch at cluster={k} dim={j}: rust={:e} vs c={:e}",
                    rust_mu[idx], c_mu_flat[idx],
                );
            }
        }
    }
}
