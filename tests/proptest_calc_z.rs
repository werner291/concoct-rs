mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1_000))]

    #[test]
    fn calc_z_matches_c(
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

        // Run mstep on each cluster to get valid parameters
        let mut all_m = vec![0.0f64; nk * nd];
        let mut all_sigma = vec![0.0f64; nk * nd * nd];
        let mut pi = vec![0.0f64; nk];
        let mut nu = vec![0.0f64; nk];
        let mut l_det = vec![0.0f64; nk];
        let mut beta = vec![0.0f64; nk];

        for k in 0..nk {
            let result = vbgmm::mstep(k, nn, nd, nk, &init_z, &data, &vb_params);
            all_m[k * nd..(k + 1) * nd].copy_from_slice(&result.m);
            all_sigma[k * nd * nd..(k + 1) * nd * nd].copy_from_slice(&result.sigma);
            pi[k] = result.pi;
            nu[k] = result.nu;
            l_det[k] = result.l_det;
            beta[k] = result.beta;
        }

        // Rust calc_z
        let mut rust_z = vec![0.0f64; nn * nk];
        vbgmm::calc_z(nn, nd, nk, &data, &mut rust_z, &all_m, &all_sigma, &pi, &nu, &l_det, &beta);

        // C calc_z: build all the pointer arrays
        let data_ptrs: Vec<*const f64> = (0..nn).map(|i| data[i * nd..].as_ptr()).collect();
        let m_ptrs: Vec<*const f64> = (0..nk).map(|k| all_m[k * nd..].as_ptr()).collect();

        // Build GSL matrices for sigma
        let sigma_mats: Vec<*mut c_ffi::GslMatrix> = (0..nk).map(|k| {
            unsafe { c_ffi::gsl_matrix_from_flat(&all_sigma[k * nd * nd..], nd) }
        }).collect();

        let mut c_z_flat = vec![0.0f64; nn * nk];
        let mut c_z_ptrs: Vec<*mut f64> = (0..nn)
            .map(|i| c_z_flat[i * nk..].as_mut_ptr())
            .collect();

        unsafe {
            c_ffi::ffi_calcZ(
                data_ptrs.as_ptr(),
                nn as i32, nk as i32, nd as i32,
                c_z_ptrs.as_ptr(),
                m_ptrs.as_ptr(),
                sigma_mats.as_ptr(),
                pi.as_ptr(), nu.as_ptr(), l_det.as_ptr(), beta.as_ptr(),
            );

            for mat in &sigma_mats { c_ffi::gsl_matrix_free(*mat); }
        }

        // Compare
        for i in 0..nn {
            for k in 0..nk {
                let idx = i * nk + k;
                assert_eq!(
                    rust_z[idx].to_bits(), c_z_flat[idx].to_bits(),
                    "z mismatch at sample={i} cluster={k}: rust={:e} vs c={:e}",
                    rust_z[idx], c_z_flat[idx],
                );
            }
        }
    }
}
