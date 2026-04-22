mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn perform_mstep_matches_c(
        n_samples in 4usize..32,
        n_dims in 2usize..6,
        n_clusters in 2usize..6,
        source_data in prop::collection::vec(-5.0f64..5.0, 32 * 6),
        source_z in prop::collection::vec(0.01f64..1.0, 32 * 6),
        source_invw0 in prop::collection::vec(-2.0f64..2.0, 6 * 6),
        beta0 in 0.001f64..1.0,
        nu0_offset in 0.0f64..5.0,
    ) {
        let nd = n_dims;
        let nn = n_samples;
        let nk = n_clusters;
        let nu0 = (nd as f64) + nu0_offset;

        let data: Vec<f64> = source_data.iter().copied().take(nn * nd).collect();
        prop_assume!(data.len() == nn * nd);
        let z: Vec<f64> = source_z.iter().copied().take(nn * nk).collect();
        prop_assume!(z.len() == nn * nk);

        let mut inv_w0 = vec![0.0f64; nd * nd];
        for i in 0..nd {
            for j in 0..nd {
                let mut sum = 0.0f64;
                for kk in 0..nd {
                    let a = if kk * nd + i < source_invw0.len() { source_invw0[kk * nd + i] } else { 0.0 };
                    let b = if kk * nd + j < source_invw0.len() { source_invw0[kk * nd + j] } else { 0.0 };
                    sum += a * b;
                }
                inv_w0[i * nd + j] = sum;
            }
        }
        for i in 0..nd { inv_w0[i * nd + i] += 1.0; }

        let vb_params = vbgmm::VBParams { beta0, nu0, inv_w0: inv_w0.clone() };

        // Rust
        let rust = vbgmm::perform_mstep(nn, nd, nk, &z, &data, &vb_params);

        // C: run mstep per cluster via FFI, then normalize pi
        let data_ptrs: Vec<*const f64> = (0..nn).map(|i| data[i * nd..].as_ptr()).collect();
        let z_ptrs: Vec<*const f64> = (0..nn).map(|i| z[i * nk..].as_ptr()).collect();
        let inv_w0_ptrs: Vec<*const f64> = (0..nd).map(|i| inv_w0[i * nd..].as_ptr()).collect();

        let mut c_pi = vec![0.0f64; nk];
        let mut c_beta = vec![0.0f64; nk];
        let mut c_nu = vec![0.0f64; nk];
        let mut c_ldet = vec![0.0f64; nk];
        let mut c_mu = vec![vec![0.0f64; nd]; nk];
        let mut c_m = vec![vec![0.0f64; nd]; nk];
        let mut c_covar = vec![vec![0.0f64; nd * nd]; nk];
        let mut c_sigma = vec![vec![0.0f64; nd * nd]; nk];

        for k in 0..nk {
            unsafe {
                c_ffi::ffi_mstep(
                    k as i32, nn as i32, nd as i32, nk as i32,
                    z_ptrs.as_ptr(), data_ptrs.as_ptr(),
                    beta0, nu0, inv_w0_ptrs.as_ptr(),
                    c_mu[k].as_mut_ptr(), c_m[k].as_mut_ptr(),
                    &mut c_pi[k], &mut c_beta[k], &mut c_nu[k], &mut c_ldet[k],
                    c_covar[k].as_mut_ptr(), c_sigma[k].as_mut_ptr(),
                );
            }
        }

        // Normalize C pi
        let c_pi_sum: f64 = c_pi.iter().sum();
        c_pi.iter_mut().for_each(|p| *p /= c_pi_sum);

        // Compare pi (the main thing this function adds over individual mstep)
        for k in 0..nk {
            assert_eq!(
                rust.pi[k].to_bits(), c_pi[k].to_bits(),
                "pi mismatch at k={k}: rust={:e} vs c={:e}", rust.pi[k], c_pi[k],
            );
        }
    }
}
