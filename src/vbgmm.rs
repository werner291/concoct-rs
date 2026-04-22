#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Constants matching c-concoct/c_vbgmm_fit.h
pub const MIN_Z: f64 = 1.0e-6;
pub const MIN_PI: f64 = 0.1;

/// VB prior parameters (matches t_VBParams).
pub struct VBParams {
    pub beta0: f64,
    pub nu0: f64,
    /// Inverse Wishart scale matrix, flat row-major, n_dims x n_dims.
    pub inv_w0: Vec<f64>,
}

/// Result of the M-step for a single component.
pub struct MStepResult {
    pub pi: f64,
    pub beta: f64,
    pub nu: f64,
    pub l_det: f64,
    /// Cluster mean, length n_dims.
    pub mu: Vec<f64>,
    /// Scaled mean (Bishop 10.61), length n_dims.
    pub m: Vec<f64>,
    /// Sample covariance matrix, flat row-major, n_dims x n_dims.
    pub covar: Vec<f64>,
    /// Inverse regularised variance (Bishop 10.62), flat row-major, n_dims x n_dims.
    pub sigma: Vec<f64>,
}

/// M-step for a single component k (Bishop 10.58-10.65).
///
/// Computes the updated parameters for component k given the current
/// responsibilities Z. Called once per component by performMStepMP.
///
/// `z` is flat row-major `[nN][nK]`: `z[i * n_clusters + k]`.
/// `data` is flat row-major `[nN][nD]`: `data[i * n_dims + j]`.
///
/// c-concoct/c_vbgmm_fit.c:530-709
pub fn mstep(
    k: usize,
    n_samples: usize,
    n_dims: usize,
    n_clusters: usize,
    z: &[f64],
    data: &[f64],
    vb_params: &VBParams,
) -> MStepResult {
    use crate::c_ffi;

    let nd = n_dims;
    let nn = n_samples;

    let mut mu = vec![0.0f64; nd];
    let mut m = vec![0.0f64; nd];
    let mut covar = vec![0.0f64; nd * nd];
    let mut inv_wk = vec![0.0f64; nd * nd];

    // Accumulate weighted mean and mixture weight
    let mut d_pi = 0.0f64;
    for i in 0..nn {
        let z_row = &z[i * n_clusters..(i + 1) * n_clusters];
        let z_ik = z_row[k];
        if z_ik > MIN_Z {
            d_pi += z_ik;
            let data_row = &data[i * nd..(i + 1) * nd];
            let mut j = 0;
            unsafe {
                let zv = _mm_set1_pd(z_ik);
                while j + 2 <= nd {
                    let d = _mm_loadu_pd(data_row.as_ptr().add(j));
                    let m = _mm_loadu_pd(mu.as_ptr().add(j));
                    let r = _mm_add_pd(m, _mm_mul_pd(zv, d));
                    _mm_storeu_pd(mu.as_mut_ptr().add(j), r);
                    j += 2;
                }
            }
            while j < nd {
                mu[j] += z_ik * data_row[j];
                j += 1;
            }
        }
    }

    let d_beta;
    let d_nu;
    let d_l_det;

    if d_pi > MIN_PI {
        // Equation 10.60
        d_beta = vb_params.beta0 + d_pi;

        for j in 0..nd {
            // Equation 10.61
            m[j] = mu[j] / d_beta;
            mu[j] /= d_pi;
        }

        d_nu = vb_params.nu0 + d_pi;

        // Calculate covariance matrices
        // diff is allocated once outside the loop (C uses a stack VLA)
        let mut diff = vec![0.0f64; nd];
        for i in 0..nn {
            let z_ik = z[i * n_clusters + k];
            if z_ik > MIN_Z {
                let data_row = &data[i * nd..(i + 1) * nd];

                // Compute diff = data_row - mu using packed SSE2 (2 at a time),
                // matching GCC's subpd codegen.
                {
                    let mut j = 0;
                    while j + 2 <= nd {
                        unsafe {
                            let d = _mm_loadu_pd(data_row.as_ptr().add(j));
                            let m = _mm_loadu_pd(mu.as_ptr().add(j));
                            let r = _mm_sub_pd(d, m);
                            _mm_storeu_pd(diff.as_mut_ptr().add(j), r);
                        }
                        j += 2;
                    }
                    // Scalar tail
                    while j < nd {
                        diff[j] = data_row[j] - mu[j];
                        j += 1;
                    }
                }

                for l in 0..nd {
                    let covar_row = &mut covar[l * nd..l * nd + nd];
                    let z_diff_l = z_ik * diff[l];

                    // Packed mul AND add — covar[l][m] and covar[l][m+1] are
                    // independent accumulators, so packed addpd is safe.
                    // Matches GCC's mulpd + addpd + movups codegen exactly.
                    let mut m_idx = 0;
                    unsafe {
                        let z_dl = _mm_set1_pd(z_diff_l);
                        while m_idx + 2 <= l + 1 {
                            let d = _mm_loadu_pd(diff.as_ptr().add(m_idx));
                            let c = _mm_loadu_pd(covar_row.as_ptr().add(m_idx));
                            let r = _mm_add_pd(c, _mm_mul_pd(z_dl, d));
                            _mm_storeu_pd(covar_row.as_mut_ptr().add(m_idx), r);
                            m_idx += 2;
                        }
                    }
                    while m_idx <= l {
                        covar_row[m_idx] += z_diff_l * diff[m_idx];
                        m_idx += 1;
                    }
                }
            }
        }

        // Symmetrise
        for l in 0..nd {
            for m_idx in (l + 1)..nd {
                covar[l * nd + m_idx] = covar[m_idx * nd + l];
            }
        }

        // Save sample covariances (covar / dPi) into the covar output
        let mut covar_out = vec![0.0f64; nd * nd];
        for l in 0..nd {
            let covar_row = &covar[l * nd..l * nd + nd];
            let out_row = &mut covar_out[l * nd..l * nd + nd];
            for m_idx in 0..nd {
                out_row[m_idx] = covar_row[m_idx] / d_pi;
            }
        }

        // Equation 10.62: InvWK = InvW0 + covar + (beta0*dPi/beta) * mu * mu^T
        let d_f = (vb_params.beta0 * d_pi) / d_beta;
        for l in 0..nd {
            let inv_w0_row = &vb_params.inv_w0[l * nd..l * nd + nd];
            let covar_row = &covar[l * nd..l * nd + nd];
            let inv_wk_row = &mut inv_wk[l * nd..l * nd + nd];
            for m_idx in 0..=l {
                inv_wk_row[m_idx] = inv_w0_row[m_idx]
                    + covar_row[m_idx]
                    + d_f * mu[l] * mu[m_idx];
            }
        }

        // Build sigma matrix (symmetric) and divide covar by dPi
        let mut sigma = vec![0.0f64; nd * nd];
        for l in 0..nd {
            let covar_row = &mut covar[l * nd..l * nd + nd];
            for m_idx in 0..=l {
                covar_row[m_idx] /= d_pi;
                sigma[l * nd + m_idx] = inv_wk[l * nd + m_idx];
                sigma[m_idx * nd + l] = inv_wk[l * nd + m_idx];
            }
        }

        // Equation 10.65
        d_l_det = {
            let mut det = (nd as f64) * (2.0f64).ln();
            for l in 0..nd {
                let dx = 0.5 * (d_nu - l as f64);
                det += unsafe { c_ffi::gsl_sf_psi(dx) };
            }
            det -= decompose_matrix(&mut sigma, nd);
            det
        };

        MStepResult {
            pi: d_pi,
            beta: d_beta,
            nu: d_nu,
            l_det: d_l_det,
            mu,
            m,
            covar: covar_out,
            sigma,
        }
    } else {
        // Empty component: reset to prior
        d_pi = 0.0;
        d_beta = vb_params.beta0;

        for j in 0..nd {
            m[j] = 0.0;
            mu[j] = 0.0;
        }

        d_nu = vb_params.nu0;

        for l in 0..nd {
            for m_idx in 0..=l {
                inv_wk[l * nd + m_idx] = vb_params.inv_w0[l * nd + m_idx];
            }
        }

        let mut sigma = vec![0.0f64; nd * nd];
        for l in 0..nd {
            for m_idx in 0..=l {
                sigma[l * nd + m_idx] = inv_wk[l * nd + m_idx];
                sigma[m_idx * nd + l] = inv_wk[l * nd + m_idx];
            }
        }

        // Equation 10.65
        d_l_det = {
            let mut det = (nd as f64) * (2.0f64).ln();
            for l in 0..nd {
                let dx = 0.5 * (d_nu - l as f64);
                det += unsafe { c_ffi::gsl_sf_psi(dx) };
            }
            det -= decompose_matrix(&mut sigma, nd);
            det
        };

        MStepResult {
            pi: d_pi,
            beta: d_beta,
            nu: d_nu,
            l_det: d_l_det,
            mu,
            m,
            covar: vec![0.0f64; nd * nd],
            sigma,
        }
    }
}

/// Euclidean distance between two vectors of equal length.
///
/// Used by k-means initialisation to assign data points to the nearest
/// cluster centroid.
///
/// c-concoct/c_vbgmm_fit.c:1300-1311
#[inline]
pub fn calc_dist(x: &[f64], mu: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), mu.len());
    let mut dist = 0.0f64;
    for i in 0..x.len() {
        let dv = x[i] - mu[i];
        dist += dv * dv;
    }
    dist.sqrt()
}

/// Per-dimension sample mean and variance from a row-major data matrix.
///
/// Used by `setVBParams` to initialise the Wishart prior (the diagonal of
/// the inverse scale matrix is `var[i] * nD`).
///
/// `data` is row-major: `data[n * n_dims + i]` is sample `n`, dimension `i`.
/// Returns `(variance, mean)` vectors of length `n_dims`.
///
/// c-concoct/c_vbgmm_fit.c:202-227
pub fn calc_sample_var(data: &[f64], n_samples: usize, n_dims: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(data.len(), n_samples * n_dims);
    let dn = n_samples as f64;

    let mut mu = vec![0.0f64; n_dims];
    let mut var = vec![0.0f64; n_dims];

    // Outer loop: dimensions. Inner loop: samples.
    // Matches C accumulation order exactly.
    for i in 0..n_dims {
        for n in 0..n_samples {
            mu[i] += data[n * n_dims + i];
            var[i] += data[n * n_dims + i] * data[n * n_dims + i];
        }

        mu[i] /= dn;

        // Variance: (sum_sq - N * mu * mu) / (N - 1)
        // Operation order: (dN * adMu[i]) * adMu[i], left-to-right.
        var[i] = (var[i] - dn * mu[i] * mu[i]) / (dn - 1.0);
    }

    (var, mu)
}

/// Cholesky decomposition, log-determinant, and in-place inversion.
///
/// Given a symmetric positive-definite matrix, computes the Cholesky
/// decomposition, extracts the log-determinant from the diagonal
/// (sum of 2*log(L[l,l])), then inverts the matrix in-place.
///
/// `matrix` is row-major, n x n. Modified in-place to hold the inverse.
/// Returns the log-determinant. Panics if Cholesky decomposition fails.
///
/// c-concoct/c_vbgmm_fit.c:507-528
pub fn decompose_matrix(matrix: &mut [f64], n: usize) -> f64 {
    use crate::c_ffi;

    unsafe {
        let gsl_m = c_ffi::gsl_matrix_from_flat(matrix, n);

        // GSL Cholesky decomposition (in-place)
        extern "C" {
            fn gsl_linalg_cholesky_decomp(m: *mut c_ffi::GslMatrix) -> i32;
            fn gsl_linalg_cholesky_invert(m: *mut c_ffi::GslMatrix) -> i32;
        }

        let status = gsl_linalg_cholesky_decomp(gsl_m);
        assert!(status != 27, "Failed Cholesky decomposition in decompose_matrix");
        // GSL_EDOM = 1, but the C code checks for it; 27 is GSL_EDOM in some versions
        // Actually let's just check != 0 for safety
        if status != 0 {
            panic!("Cholesky decomposition failed with status {status}");
        }

        // Log-determinant: sum of 2*log(diagonal)
        let mut det = 0.0f64;
        for l in 0..n {
            let dt = c_ffi::gsl_matrix_get(gsl_m, l, l);
            det += 2.0 * dt.ln();
        }

        // Invert in-place
        gsl_linalg_cholesky_invert(gsl_m);

        // Copy result back
        let result = c_ffi::gsl_matrix_to_flat(gsl_m, n);
        matrix.copy_from_slice(&result);

        c_ffi::gsl_matrix_free(gsl_m);

        det
    }
}

/// Log Wishart normalisation constant.
///
/// Computes the log of the normalisation constant B for a Wishart
/// distribution. Used in the variational lower bound calculation.
/// `b_inv`: if true, the input is the inverse scale matrix.
///
/// c-concoct/c_vbgmm_fit.c:1207-1236
pub fn d_log_wishart_b(matrix: &[f64], n: usize, nu: f64, b_inv: bool) -> f64 {
    use crate::c_ffi;
    let d = n as f64;

    unsafe {
        // Copy matrix — decomposeMatrix modifies in-place
        let gsl_m = c_ffi::gsl_matrix_from_flat(matrix, n);
        let log_det = decompose_matrix_gsl(gsl_m, n);

        let ret = if b_inv {
            0.5 * nu * log_det
        } else {
            -0.5 * nu * log_det
        };

        let mut t = 0.5 * nu * d * (2.0f64).ln();
        t += 0.25 * d * (d - 1.0) * std::f64::consts::PI.ln();

        for i in 0..n {
            t += c_ffi::gsl_sf_lngamma(0.5 * (nu - i as f64));
        }

        c_ffi::gsl_matrix_free(gsl_m);
        ret - t
    }
}

/// Expected log-determinant of a Wishart distribution.
///
/// Note: modifies `matrix` in-place via decomposeMatrix (Cholesky + invert).
///
/// c-concoct/c_vbgmm_fit.c:1238-1257
pub fn d_wishart_expect_log_det(matrix: &mut [f64], n: usize, nu: f64) -> f64 {
    use crate::c_ffi;
    let d = n as f64;

    // The C code allocates a copy but then decomposes the original.
    // We match that behaviour: decompose matrix in-place.
    let log_det = decompose_matrix(matrix, n);

    let mut ret = d * (2.0f64).ln() + log_det;

    for i in 0..n {
        ret += unsafe { c_ffi::gsl_sf_psi(0.5 * (nu - i as f64)) };
    }

    ret
}

/// Internal: call decomposeMatrix on a GSL matrix directly.
/// Used by d_log_wishart_b which needs GSL matrix allocation anyway.
unsafe fn decompose_matrix_gsl(m: *mut crate::c_ffi::GslMatrix, n: usize) -> f64 {
    use crate::c_ffi;
    extern "C" {
        fn gsl_linalg_cholesky_decomp(m: *mut crate::c_ffi::GslMatrix) -> i32;
        fn gsl_linalg_cholesky_invert(m: *mut crate::c_ffi::GslMatrix) -> i32;
    }

    let status = gsl_linalg_cholesky_decomp(m);
    assert!(status == 0, "Cholesky decomposition failed with status {status}");

    let mut det = 0.0f64;
    for l in 0..n {
        let dt = c_ffi::gsl_matrix_get(m, l, l);
        det += 2.0 * dt.ln();
    }

    gsl_linalg_cholesky_invert(m);
    det
}

/// Recompute cluster centroids from hard assignments.
///
/// For each cluster k, the mean is the average of all data points assigned
/// to it. Empty clusters get a zero mean.
///
/// `data` is row-major: `data[n * n_dims + j]` is sample n, dimension j.
/// `assignments[n]` is the cluster index for sample n.
/// `weights[k]` is the number of samples assigned to cluster k.
/// `mu` is caller-allocated output, length `n_clusters * n_dims` (row-major).
///
/// c-concoct/c_vbgmm_fit.c:1259-1298
pub fn update_means(
    data: &[f64],
    n_samples: usize,
    n_clusters: usize,
    n_dims: usize,
    assignments: &[i32],
    weights: &[i32],
    mu: &mut [f64],
) {
    debug_assert_eq!(data.len(), n_samples * n_dims);
    debug_assert_eq!(assignments.len(), n_samples);
    debug_assert_eq!(weights.len(), n_clusters);
    debug_assert_eq!(mu.len(), n_clusters * n_dims);

    // Phase 1: zero.
    for v in mu.iter_mut() {
        *v = 0.0;
    }

    // Phase 2: accumulate. Outer loop over samples, inner over dims.
    for i in 0..n_samples {
        let nz = assignments[i] as usize;
        let data_row = &data[i * n_dims..i * n_dims + n_dims];
        let mu_row = &mut mu[nz * n_dims..nz * n_dims + n_dims];
        for j in 0..n_dims {
            mu_row[j] += data_row[j];
        }
    }

    // Phase 3: normalise.
    for k in 0..n_clusters {
        let mu_row = &mut mu[k * n_dims..k * n_dims + n_dims];
        if weights[k] > 0 {
            let w = weights[k] as f64;
            for j in 0..n_dims {
                mu_row[j] /= w;
            }
        } else {
            for j in 0..n_dims {
                mu_row[j] = 0.0;
            }
        }
    }
}
