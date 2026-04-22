mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn calc_vbl_matches_c(
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
        let init_z: Vec<f64> = source_z.iter().copied().take(nn * nk).collect();
        prop_assume!(init_z.len() == nn * nk);

        // Build SPD prior
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
        let log_wishart_b = vbgmm::d_log_wishart_b(&inv_w0, nd, nu0, true);

        // Run mstep per cluster to get valid state
        let mut all_mu = vec![0.0f64; nk * nd];
        let mut all_m = vec![0.0f64; nk * nd];
        let mut all_covar = vec![0.0f64; nk * nd * nd];
        let mut all_sigma = vec![0.0f64; nk * nd * nd];
        let mut pi_v = vec![0.0f64; nk];
        let mut beta_v = vec![0.0f64; nk];
        let mut nu_v = vec![0.0f64; nk];
        let mut l_det_v = vec![0.0f64; nk];

        for k in 0..nk {
            let r = vbgmm::mstep(k, nn, nd, nk, &init_z, &data, &vb_params);
            all_mu[k * nd..(k + 1) * nd].copy_from_slice(&r.mu);
            all_m[k * nd..(k + 1) * nd].copy_from_slice(&r.m);
            all_covar[k * nd * nd..(k + 1) * nd * nd].copy_from_slice(&r.covar);
            all_sigma[k * nd * nd..(k + 1) * nd * nd].copy_from_slice(&r.sigma);
            pi_v[k] = r.pi;
            beta_v[k] = r.beta;
            nu_v[k] = r.nu;
            l_det_v[k] = r.l_det;
        }

        // Normalize pi
        let pi_sum: f64 = pi_v.iter().sum();
        if pi_sum > 0.0 {
            pi_v.iter_mut().for_each(|p| *p /= pi_sum);
        }

        // Rust
        let mut rust_sigma = all_sigma.clone();
        let rust_vbl = vbgmm::calc_vbl(
            nn, nd, nk, &init_z, &all_mu, &all_m, &all_covar, &mut rust_sigma,
            &pi_v, &beta_v, &nu_v, &l_det_v, &inv_w0, beta0, nu0, log_wishart_b,
        );

        // C: build pointer arrays
        let data_ptrs: Vec<*const f64> = (0..nn).map(|i| data[i * nd..].as_ptr()).collect();
        let z_ptrs: Vec<*const f64> = (0..nn).map(|i| init_z[i * nk..].as_ptr()).collect();
        let mu_ptrs: Vec<*const f64> = (0..nk).map(|k| all_mu[k * nd..].as_ptr()).collect();
        let m_ptrs: Vec<*const f64> = (0..nk).map(|k| all_m[k * nd..].as_ptr()).collect();

        let mut c_sigma = all_sigma.clone();
        let covar_mats: Vec<*mut c_ffi::GslMatrix> = (0..nk).map(|k| {
            unsafe { c_ffi::gsl_matrix_from_flat(&all_covar[k * nd * nd..], nd) }
        }).collect();
        let sigma_mats: Vec<*mut c_ffi::GslMatrix> = (0..nk).map(|k| {
            unsafe { c_ffi::gsl_matrix_from_flat(&c_sigma[k * nd * nd..], nd) }
        }).collect();
        let gsl_invw0 = unsafe { c_ffi::gsl_matrix_from_flat(&inv_w0, nd) };

        let c_vbl = unsafe {
            c_ffi::ffi_calcVBL(
                data_ptrs.as_ptr(),
                nn as i32, nk as i32, nd as i32,
                z_ptrs.as_ptr(),
                mu_ptrs.as_ptr(),
                m_ptrs.as_ptr(),
                covar_mats.as_ptr(),
                sigma_mats.as_ptr(),
                pi_v.as_ptr(), beta_v.as_ptr(), nu_v.as_ptr(), l_det_v.as_ptr(),
                beta0, nu0, gsl_invw0, log_wishart_b,
            )
        };

        unsafe {
            for mat in &covar_mats { c_ffi::gsl_matrix_free(*mat); }
            for mat in &sigma_mats { c_ffi::gsl_matrix_free(*mat); }
            c_ffi::gsl_matrix_free(gsl_invw0);
        }

        assert_eq!(
            rust_vbl.to_bits(), c_vbl.to_bits(),
            "VBL mismatch: rust={rust_vbl:e} vs c={c_vbl:e}",
        );
    }
}
