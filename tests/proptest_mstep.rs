mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2_000))]

    #[test]
    fn mstep_matches_c(
        n_samples in 4usize..64,
        n_dims in 2usize..8,
        n_clusters in 2usize..8,
        k_idx in 0usize..8,
        source_data in prop::collection::vec(-5.0f64..5.0, 64 * 8),
        source_z in prop::collection::vec(0.0f64..1.0, 64 * 8),
        source_invw0 in prop::collection::vec(-3.0f64..3.0, 8 * 8),
        beta0 in 0.001f64..1.0,
        nu0_offset in 0.0f64..5.0,
    ) {
        let k = k_idx % n_clusters;
        let nd = n_dims;
        let nn = n_samples;
        let nk = n_clusters;
        let nu0 = (nd as f64) + nu0_offset;

        // Build data (flat row-major)
        let data: Vec<f64> = source_data.iter().copied().take(nn * nd).collect();
        prop_assume!(data.len() == nn * nd);

        // Build responsibilities (flat row-major, nn x nk)
        let z: Vec<f64> = source_z.iter().copied().take(nn * nk).collect();
        prop_assume!(z.len() == nn * nk);

        // Build SPD prior matrix: A^T*A + I
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

        let vb_params = vbgmm::VBParams {
            beta0,
            nu0,
            inv_w0: inv_w0.clone(),
        };

        // Rust
        let rust = vbgmm::mstep(k, nn, nd, nk, &z, &data, &vb_params);

        // C: build row-pointer arrays
        let data_ptrs: Vec<*const f64> = (0..nn).map(|i| data[i * nd..].as_ptr()).collect();
        let z_ptrs: Vec<*const f64> = (0..nn).map(|i| z[i * nk..].as_ptr()).collect();
        let inv_w0_ptrs: Vec<*const f64> = (0..nd).map(|i| inv_w0[i * nd..].as_ptr()).collect();

        let mut c_mu = vec![0.0f64; nd];
        let mut c_m = vec![0.0f64; nd];
        let mut c_pi = 0.0f64;
        let mut c_beta = 0.0f64;
        let mut c_nu = 0.0f64;
        let mut c_ldet = 0.0f64;
        let mut c_covar = vec![0.0f64; nd * nd];
        let mut c_sigma = vec![0.0f64; nd * nd];

        unsafe {
            c_ffi::ffi_mstep(
                k as i32, nn as i32, nd as i32, nk as i32,
                z_ptrs.as_ptr(), data_ptrs.as_ptr(),
                beta0, nu0, inv_w0_ptrs.as_ptr(),
                c_mu.as_mut_ptr(), c_m.as_mut_ptr(),
                &mut c_pi, &mut c_beta, &mut c_nu, &mut c_ldet,
                c_covar.as_mut_ptr(), c_sigma.as_mut_ptr(),
            );
        }

        // Compare scalars
        assert_eq!(rust.pi.to_bits(), c_pi.to_bits(), "pi mismatch");
        assert_eq!(rust.beta.to_bits(), c_beta.to_bits(), "beta mismatch");
        assert_eq!(rust.nu.to_bits(), c_nu.to_bits(), "nu mismatch");
        assert_eq!(rust.l_det.to_bits(), c_ldet.to_bits(), "l_det mismatch");

        // Compare vectors
        for j in 0..nd {
            assert_eq!(rust.mu[j].to_bits(), c_mu[j].to_bits(), "mu[{j}] mismatch");
            assert_eq!(rust.m[j].to_bits(), c_m[j].to_bits(), "m[{j}] mismatch");
        }

        // Compare matrices
        for idx in 0..nd * nd {
            assert_eq!(rust.covar[idx].to_bits(), c_covar[idx].to_bits(),
                "covar[{}] mismatch", idx);
            assert_eq!(rust.sigma[idx].to_bits(), c_sigma[idx].to_bits(),
                "sigma[{}] mismatch", idx);
        }
    }
}
