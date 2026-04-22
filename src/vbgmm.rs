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

/// Scalar covariance accumulation loop.
fn covar_accum_scalar(
    z: &[f64], data: &[f64], mu: &[f64],
    nd: usize, nk: usize, k: usize, covar: &mut [f64],
) {
    let mut diff = vec![0.0f64; nd];
    for (z_row, data_row) in z.chunks_exact(nk).zip(data.chunks_exact(nd)) {
        let z_ik = z_row[k];
        if z_ik > MIN_Z {
            diff.iter_mut().zip(data_row.iter().zip(mu))
                .for_each(|(d, (&x, &m))| *d = x - m);

            for l in 0..nd {
                let covar_row = &mut covar[l * nd..l * nd + nd];
                let z_diff_l = z_ik * diff[l];
                for m_idx in 0..=l {
                    covar_row[m_idx] += z_diff_l * diff[m_idx];
                }
            }
        }
    }
}

/// AVX2 covariance accumulation: 4 doubles per iteration.
/// Produces bit-identical results to scalar (each covar[l][m] is an
/// independent accumulator, so packed add is safe).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn covar_accum_avx2(
    z: &[f64], data: &[f64], mu: &[f64],
    nd: usize, nk: usize, k: usize, covar: &mut [f64],
) {
    use std::arch::x86_64::*;

    let mut diff = vec![0.0f64; nd];
    for (z_row, data_row) in z.chunks_exact(nk).zip(data.chunks_exact(nd)) {
        let z_ik = z_row[k];
        if z_ik > MIN_Z {
            // diff = data_row - mu, 4 at a time
            let mut j = 0;
            while j + 4 <= nd {
                let d = _mm256_loadu_pd(data_row.as_ptr().add(j));
                let m = _mm256_loadu_pd(mu.as_ptr().add(j));
                _mm256_storeu_pd(diff.as_mut_ptr().add(j), _mm256_sub_pd(d, m));
                j += 4;
            }
            while j < nd { diff[j] = data_row[j] - mu[j]; j += 1; }

            for l in 0..nd {
                let covar_row = &mut covar[l * nd..l * nd + nd];
                let z_diff_l = z_ik * diff[l];
                let zd = _mm256_set1_pd(z_diff_l);
                let mut m_idx = 0;
                while m_idx + 4 <= l + 1 {
                    let d = _mm256_loadu_pd(diff.as_ptr().add(m_idx));
                    let c = _mm256_loadu_pd(covar_row.as_ptr().add(m_idx));
                    _mm256_storeu_pd(
                        covar_row.as_mut_ptr().add(m_idx),
                        _mm256_add_pd(c, _mm256_mul_pd(zd, d)),
                    );
                    m_idx += 4;
                }
                while m_idx <= l {
                    covar_row[m_idx] += z_diff_l * diff[m_idx];
                    m_idx += 1;
                }
            }
        }
    }
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
///
/// Note: an SSE2 intrinsics version existed in a prior commit to investigate
/// a codegen gap between GCC and LLVM. This idiomatic version produces
/// bit-identical results (verified by proptest) and is preferred for
/// readability and safety. See notes/codegen-mstep.md.
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

    let mut mu = vec![0.0f64; nd];
    let mut d_pi = 0.0f64;

    for (z_row, data_row) in z.chunks_exact(n_clusters).zip(data.chunks_exact(nd)) {
        let z_ik = z_row[k];
        if z_ik > MIN_Z {
            d_pi += z_ik;
            mu.iter_mut().zip(data_row).for_each(|(m, &d)| *m += z_ik * d);
        }
    }

    if d_pi > MIN_PI {
        let d_beta = vb_params.beta0 + d_pi;
        let d_nu = vb_params.nu0 + d_pi;

        let mut m = vec![0.0f64; nd];
        for j in 0..nd {
            m[j] = mu[j] / d_beta;
            mu[j] /= d_pi;
        }

        // Covariance
        let mut covar = vec![0.0f64; nd * nd];
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 detected at runtime. The AVX2 path produces
                // bit-identical results to scalar (independent accumulators).
                unsafe {
                    covar_accum_avx2(&z, &data, &mu, nd, n_clusters, k, &mut covar);
                }
            } else {
                covar_accum_scalar(&z, &data, &mu, nd, n_clusters, k, &mut covar);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            covar_accum_scalar(&z, &data, &mu, nd, n_clusters, k, &mut covar);
        }

        // Symmetrise
        for l in 0..nd {
            for m_idx in (l + 1)..nd {
                covar[l * nd + m_idx] = covar[m_idx * nd + l];
            }
        }

        let mut covar_out = vec![0.0f64; nd * nd];
        covar_out.iter_mut().zip(&covar).for_each(|(o, &c)| *o = c / d_pi);

        // Eq 10.62
        let d_f = (vb_params.beta0 * d_pi) / d_beta;
        let mut sigma = vec![0.0f64; nd * nd];
        for l in 0..nd {
            for m_idx in 0..=l {
                let inv_wk = vb_params.inv_w0[l * nd + m_idx]
                    + covar[l * nd + m_idx]
                    + d_f * mu[l] * mu[m_idx];
                sigma[l * nd + m_idx] = inv_wk;
                sigma[m_idx * nd + l] = inv_wk;
            }
        }

        // Eq 10.65
        let mut d_l_det = (nd as f64) * 2.0f64.ln();
        for l in 0..nd {
            d_l_det += unsafe { c_ffi::gsl_sf_psi(0.5 * (d_nu - l as f64)) };
        }
        d_l_det -= decompose_matrix(&mut sigma, nd);

        MStepResult { pi: d_pi, beta: d_beta, nu: d_nu, l_det: d_l_det, mu, m, covar: covar_out, sigma }
    } else {
        let d_beta = vb_params.beta0;
        let d_nu = vb_params.nu0;
        let m = vec![0.0f64; nd];
        mu.fill(0.0);

        let mut sigma = vec![0.0f64; nd * nd];
        for l in 0..nd {
            for m_idx in 0..=l {
                let v = vb_params.inv_w0[l * nd + m_idx];
                sigma[l * nd + m_idx] = v;
                sigma[m_idx * nd + l] = v;
            }
        }

        let mut d_l_det = (nd as f64) * 2.0f64.ln();
        for l in 0..nd {
            d_l_det += unsafe { c_ffi::gsl_sf_psi(0.5 * (d_nu - l as f64)) };
        }
        d_l_det -= decompose_matrix(&mut sigma, nd);

        MStepResult { pi: 0.0, beta: d_beta, nu: d_nu, l_det: d_l_det, mu, m, covar: vec![0.0f64; nd*nd], sigma }
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

/// E-step: compute responsibilities Z for all data points.
///
/// For each sample i and cluster k, computes the (unnormalized) probability
/// that sample i belongs to cluster k, then normalizes. Uses the Mahalanobis
/// distance via GSL dsymv/ddot.
///
/// All flat arrays are row-major.
/// `z`: output, `[n_samples][n_clusters]`.
/// `data`: `[n_samples][n_dims]`.
/// `m`: scaled means, `[n_clusters][n_dims]`.
/// `sigma`: inverse regularised variances, `[n_clusters][n_dims * n_dims]`.
/// `pi`, `nu`, `l_det`, `beta`: per-cluster parameters, length `n_clusters`.
///
/// c-concoct/c_vbgmm_fit.c:979-1046
pub fn calc_z(
    n_samples: usize,
    n_dims: usize,
    n_clusters: usize,
    data: &[f64],
    z: &mut [f64],
    m: &[f64],
    sigma: &[f64],
    pi: &[f64],
    nu: &[f64],
    l_det: &[f64],
    beta: &[f64],
) {
    use crate::c_ffi;

    let dd = n_dims as f64;
    let nk = n_clusters;
    let nd = n_dims;

    // CblasLower = 122 in GSL
    const CBLAS_LOWER: i32 = 122;

    for i in 0..n_samples {
        let data_row = &data[i * nd..(i + 1) * nd];
        let z_row = &mut z[i * nk..(i + 1) * nk];

        let mut dist = vec![0.0f64; nk];
        let mut d_min_dist = f64::MAX;

        // Phase 1: compute distance for each active cluster
        for k in 0..nk {
            if pi[k] > 0.0 {
                unsafe {
                    let pt_diff = c_ffi::gsl_vector_alloc(nd);
                    let pt_res = c_ffi::gsl_vector_alloc(nd);

                    // diff = data[i] - m[k]
                    let m_row = &m[k * nd..(k + 1) * nd];
                    for l in 0..nd {
                        c_ffi::gsl_vector_set(pt_diff, l, data_row[l] - m_row[l]);
                    }

                    // Build GSL matrix view for sigma[k]
                    let sigma_k = &sigma[k * nd * nd..(k + 1) * nd * nd];
                    let gsl_sigma = c_ffi::gsl_matrix_from_flat(sigma_k, nd);

                    // res = sigma[k] * diff  (symmetric matrix-vector product)
                    c_ffi::gsl_blas_dsymv(CBLAS_LOWER, 1.0, gsl_sigma, pt_diff, 0.0, pt_res);

                    // dist[k] = diff^T * res
                    c_ffi::gsl_blas_ddot(pt_diff, pt_res, &mut dist[k]);

                    c_ffi::gsl_matrix_free(gsl_sigma);
                    c_ffi::gsl_vector_free(pt_res);
                    c_ffi::gsl_vector_free(pt_diff);
                }

                // Scale and shift — order matches C exactly
                dist[k] *= nu[k];
                dist[k] -= l_det[k];
                dist[k] += dd / beta[k];

                if dist[k] < d_min_dist {
                    d_min_dist = dist[k];
                }
            }
        }

        // Phase 2: unnormalized responsibilities
        let mut d_total_z = 0.0f64;
        for k in 0..nk {
            if pi[k] > 0.0 {
                z_row[k] = pi[k] * (-0.5 * (dist[k] - d_min_dist)).exp();
                d_total_z += z_row[k];
            } else {
                z_row[k] = 0.0;
            }
        }

        // Phase 3: zero tiny responsibilities, recompute total
        let mut d_n_total_z = 0.0f64;
        for k in 0..nk {
            let d_f = z_row[k] / d_total_z;
            if d_f < MIN_Z {
                z_row[k] = 0.0;
            }
            d_n_total_z += z_row[k];
        }

        // Phase 4: normalize
        if d_n_total_z > 0.0 {
            for k in 0..nk {
                z_row[k] /= d_n_total_z;
            }
        }
    }
}

/// Bishop Equation 10.71 term for one component.
/// c-concoct/c_vbgmm_fit.c:824-857
fn eqn_a(
    nd: usize, covar_k: &[f64], sigma_k: &[f64],
    mu_k: &[f64], m_k: &[f64],
    l_det_k: f64, nu_k: f64, log_d2pi: f64, beta_k: f64, n_k: f64,
) -> f64 {
    use crate::c_ffi;
    let dd = nd as f64;

    unsafe {
        let gsl_covar = c_ffi::gsl_matrix_from_flat(covar_k, nd);
        let gsl_sigma = c_ffi::gsl_matrix_from_flat(sigma_k, nd);
        let gsl_res = c_ffi::gsl_matrix_alloc(nd, nd);

        // res = covar * sigma
        const CBLAS_NO_TRANS: i32 = 111;
        c_ffi::gsl_blas_dgemm(CBLAS_NO_TRANS, CBLAS_NO_TRANS, 1.0,
                              gsl_covar, gsl_sigma, 0.0, gsl_res);

        // trace(res)
        let mut dt1 = 0.0f64;
        for l in 0..nd {
            dt1 += c_ffi::gsl_matrix_get(gsl_res, l, l);
        }

        // diff = mu_k - m_k, then dt2 = diff^T * sigma * diff
        let pt_diff = c_ffi::gsl_vector_alloc(nd);
        let pt_r = c_ffi::gsl_vector_alloc(nd);
        for l in 0..nd {
            c_ffi::gsl_vector_set(pt_diff, l, mu_k[l] - m_k[l]);
        }
        const CBLAS_LOWER: i32 = 122;
        c_ffi::gsl_blas_dsymv(CBLAS_LOWER, 1.0, gsl_sigma, pt_diff, 0.0, pt_r);
        let mut dt2 = 0.0f64;
        c_ffi::gsl_blas_ddot(pt_diff, pt_r, &mut dt2);

        let d_f = l_det_k - nu_k * (dt1 + dt2) - dd * (log_d2pi + (1.0 / beta_k));
        let ret = 0.5 * n_k * d_f;

        c_ffi::gsl_matrix_free(gsl_res);
        c_ffi::gsl_matrix_free(gsl_covar);
        c_ffi::gsl_matrix_free(gsl_sigma);
        c_ffi::gsl_vector_free(pt_diff);
        c_ffi::gsl_vector_free(pt_r);

        ret
    }
}

/// Bishop Equation 10.74 term for one component.
/// c-concoct/c_vbgmm_fit.c:859-892
fn eqn_b(
    nd: usize, inv_w0: &[f64], sigma_k: &[f64],
    m_k: &[f64], beta0: f64, d2pi: f64,
    l_det_k: f64, beta_k: f64, nu_k: f64, nu0: f64,
) -> f64 {
    use crate::c_ffi;
    let dd = nd as f64;

    unsafe {
        let gsl_invw0 = c_ffi::gsl_matrix_from_flat(inv_w0, nd);
        let gsl_sigma = c_ffi::gsl_matrix_from_flat(sigma_k, nd);
        let gsl_res = c_ffi::gsl_matrix_alloc(nd, nd);

        const CBLAS_NO_TRANS: i32 = 111;
        c_ffi::gsl_blas_dgemm(CBLAS_NO_TRANS, CBLAS_NO_TRANS, 1.0,
                              gsl_invw0, gsl_sigma, 0.0, gsl_res);

        let mut dt1 = 0.0f64;
        for l in 0..nd {
            dt1 += c_ffi::gsl_matrix_get(gsl_res, l, l);
        }

        let pt_diff = c_ffi::gsl_vector_alloc(nd);
        let pt_r = c_ffi::gsl_vector_alloc(nd);
        for l in 0..nd {
            c_ffi::gsl_vector_set(pt_diff, l, m_k[l]);
        }
        const CBLAS_LOWER: i32 = 122;
        c_ffi::gsl_blas_dsymv(CBLAS_LOWER, 1.0, gsl_sigma, pt_diff, 0.0, pt_r);
        let mut dt2 = 0.0f64;
        c_ffi::gsl_blas_ddot(pt_diff, pt_r, &mut dt2);

        let d_f = dd * (beta0 / d2pi).ln() + l_det_k
            - ((dd * beta0) / beta_k)
            - beta0 * nu_k * dt2
            - nu_k * dt1;
        let ret = 0.5 * (d_f + (nu0 - dd - 1.0) * l_det_k);

        c_ffi::gsl_matrix_free(gsl_res);
        c_ffi::gsl_matrix_free(gsl_invw0);
        c_ffi::gsl_matrix_free(gsl_sigma);
        c_ffi::gsl_vector_free(pt_diff);
        c_ffi::gsl_vector_free(pt_r);

        ret
    }
}

/// Variational lower bound (Bishop 10.71-10.77).
///
/// All flat arrays row-major.
/// `z`: `[n_samples][n_clusters]`.
/// `mu`, `m`: `[n_clusters][n_dims]`.
/// `covar`, `sigma`: `[n_clusters][n_dims * n_dims]`.
/// `pi`, `beta`, `nu`, `l_det`: per-cluster, length `n_clusters`.
/// `inv_w0`: prior inverse Wishart scale, `[n_dims * n_dims]`.
/// `log_wishart_b`: precomputed log Wishart B from setVBParams.
///
/// c-concoct/c_vbgmm_fit.c:895-977
pub fn calc_vbl(
    n_samples: usize,
    n_dims: usize,
    n_clusters: usize,
    z: &[f64],
    mu: &[f64],
    m: &[f64],
    covar: &[f64],
    sigma: &mut [f64],
    pi: &[f64],
    beta: &[f64],
    nu: &[f64],
    l_det: &[f64],
    inv_w0: &[f64],
    beta0: f64,
    nu0: f64,
    log_wishart_b: f64,
) -> f64 {
    let nn = n_samples;
    let nk = n_clusters;
    let nd = n_dims;
    let dd = nd as f64;
    let d2pi = 2.0 * std::f64::consts::PI;
    let log_d2pi = d2pi.ln();

    // Compute N_k and Bishop2 (Eq 10.72)
    let mut n_k = vec![0.0f64; nk];
    let mut bishop2 = 0.0f64;
    for i in 0..nn {
        let z_row = &z[i * nk..(i + 1) * nk];
        for k in 0..nk {
            n_k[k] += z_row[k];
            if pi[k] > 0.0 {
                bishop2 += z_row[k] * pi[k].ln();
            }
        }
    }

    let mut d_k = 0.0f64;
    for k in 0..nk {
        if n_k[k] > 0.0 { d_k += 1.0; }
    }

    // Bishop1 (Eq 10.71)
    let mut bishop1 = 0.0f64;
    for k in 0..nk {
        if n_k[k] > 0.0 {
            bishop1 += eqn_a(nd,
                &covar[k * nd * nd..(k + 1) * nd * nd],
                &sigma[k * nd * nd..(k + 1) * nd * nd],
                &mu[k * nd..(k + 1) * nd],
                &m[k * nd..(k + 1) * nd],
                l_det[k], nu[k], log_d2pi, beta[k], n_k[k]);
        }
    }

    // Bishop3 (Eq 10.74)
    let mut bishop3 = 0.0f64;
    for k in 0..nk {
        if n_k[k] > 0.0 {
            bishop3 += eqn_b(nd, inv_w0,
                &sigma[k * nd * nd..(k + 1) * nd * nd],
                &m[k * nd..(k + 1) * nd],
                beta0, d2pi, l_det[k], beta[k], nu[k], nu0);
        }
    }
    bishop3 += d_k * log_wishart_b;

    // Bishop4 (Eq 10.75)
    let mut bishop4 = 0.0f64;
    for i in 0..nn {
        let z_row = &z[i * nk..(i + 1) * nk];
        for k in 0..nk {
            if z_row[k] > 0.0 {
                bishop4 += z_row[k] * z_row[k].ln();
            }
        }
    }

    // Bishop5 (Eq 10.77)
    let mut bishop5 = 0.0f64;
    for k in 0..nk {
        if n_k[k] > 0.0 {
            let mut sigma_k = sigma[k * nd * nd..(k + 1) * nd * nd].to_vec();
            bishop5 += 0.5 * l_det[k]
                + 0.5 * dd * (beta[k] / d2pi).ln()
                - 0.5 * dd
                - d_wishart_expect_log_det(&mut sigma_k, nd, nu[k]);
            sigma[k * nd * nd..(k + 1) * nd * nd].copy_from_slice(&sigma_k);
        }
    }

    bishop1 + bishop2 + bishop3 - bishop4 - bishop5
}
