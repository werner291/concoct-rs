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
    use rayon::prelude::*;

    let dd = n_dims as f64;
    let nk = n_clusters;
    let nd = n_dims;

    use crate::c_ffi;
    use rayon::prelude::*;
    const CBLAS_LOWER: i32 = 122;

    // Pre-build GSL sigma matrices (read-only, shared across threads).
    // Wrap raw pointers for Send — GSL dsymv only reads the matrix.
    struct SendPtr(*mut c_ffi::GslMatrix);
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}

    let gsl_sigmas: Vec<SendPtr> = (0..nk).map(|k| {
        SendPtr(unsafe { c_ffi::gsl_matrix_from_flat(&sigma[k * nd * nd..], nd) })
    }).collect();

    // Parallel over samples — same as C's #pragma omp parallel for
    z.par_chunks_mut(nk)
        .zip(data.par_chunks(nd))
        .for_each(|(z_row, data_row)| {
            // Per-thread GSL vectors (same as C's per-OMP-thread alloc)
            let pt_diff = unsafe { c_ffi::gsl_vector_alloc(nd) };
            let pt_res = unsafe { c_ffi::gsl_vector_alloc(nd) };

            let mut dist = vec![0.0f64; nk];
            let mut d_min_dist = f64::MAX;

            for k in 0..nk {
                if pi[k] > 0.0 {
                    unsafe {
                        let m_row = &m[k * nd..(k + 1) * nd];
                        for l in 0..nd {
                            c_ffi::gsl_vector_set(pt_diff, l, data_row[l] - m_row[l]);
                        }

                        c_ffi::gsl_blas_dsymv(CBLAS_LOWER, 1.0, gsl_sigmas[k].0,
                                              pt_diff, 0.0, pt_res);
                        c_ffi::gsl_blas_ddot(pt_diff, pt_res, &mut dist[k]);
                    }

                    dist[k] *= nu[k];
                    dist[k] -= l_det[k];
                    dist[k] += dd / beta[k];

                    if dist[k] < d_min_dist {
                        d_min_dist = dist[k];
                    }
                }
            }

            let mut d_total_z = 0.0f64;
            for k in 0..nk {
                if pi[k] > 0.0 {
                    z_row[k] = pi[k] * (-0.5 * (dist[k] - d_min_dist)).exp();
                    d_total_z += z_row[k];
                } else {
                    z_row[k] = 0.0;
                }
            }

            let mut d_n_total_z = 0.0f64;
            for k in 0..nk {
                let d_f = z_row[k] / d_total_z;
                if d_f < MIN_Z {
                    z_row[k] = 0.0;
                }
                d_n_total_z += z_row[k];
            }

            if d_n_total_z > 0.0 {
                for k in 0..nk {
                    z_row[k] /= d_n_total_z;
                }
            }

            unsafe {
                c_ffi::gsl_vector_free(pt_res);
                c_ffi::gsl_vector_free(pt_diff);
            }
        });

    // Free shared sigma matrices
    unsafe {
        for s in &gsl_sigmas { c_ffi::gsl_matrix_free(s.0); }
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

/// Result of performing the M-step across all components.
pub struct PerformMStepResult {
    pub mu: Vec<f64>,       // [nK][nD]
    pub m: Vec<f64>,        // [nK][nD]
    pub covar: Vec<f64>,    // [nK][nD*nD]
    pub sigma: Vec<f64>,    // [nK][nD*nD]
    pub pi: Vec<f64>,       // [nK]
    pub beta: Vec<f64>,     // [nK]
    pub nu: Vec<f64>,       // [nK]
    pub l_det: Vec<f64>,    // [nK]
}

/// M-step across all components, then normalize pi.
///
/// Calls mstep per cluster (sequential — the C uses OMP parallel for, but
/// the accumulation within each cluster is independent, so sequential
/// produces the same result). Then normalizes pi by dividing by the sum.
///
/// c-concoct/c_vbgmm_fit.c:711-751
pub fn perform_mstep(
    n_samples: usize,
    n_dims: usize,
    n_clusters: usize,
    z: &[f64],
    data: &[f64],
    vb_params: &VBParams,
) -> PerformMStepResult {
    use rayon::prelude::*;

    let nd = n_dims;
    let nk = n_clusters;

    // Parallel over clusters — same as C's #pragma omp parallel for.
    // Each mstep(k) is independent: reads shared z/data, writes only to
    // its own output. No shared accumulators, so parallelism doesn't
    // change float results.
    let results: Vec<MStepResult> = (0..nk)
        .into_par_iter()
        .map(|k| mstep(k, n_samples, nd, nk, z, data, vb_params))
        .collect();

    // Scatter into flat arrays
    let mut all_mu = vec![0.0f64; nk * nd];
    let mut all_m = vec![0.0f64; nk * nd];
    let mut all_covar = vec![0.0f64; nk * nd * nd];
    let mut all_sigma = vec![0.0f64; nk * nd * nd];
    let mut pi_v = vec![0.0f64; nk];
    let mut beta_v = vec![0.0f64; nk];
    let mut nu_v = vec![0.0f64; nk];
    let mut l_det_v = vec![0.0f64; nk];

    for (k, r) in results.into_iter().enumerate() {
        all_mu[k * nd..(k + 1) * nd].copy_from_slice(&r.mu);
        all_m[k * nd..(k + 1) * nd].copy_from_slice(&r.m);
        all_covar[k * nd * nd..(k + 1) * nd * nd].copy_from_slice(&r.covar);
        all_sigma[k * nd * nd..(k + 1) * nd * nd].copy_from_slice(&r.sigma);
        pi_v[k] = r.pi;
        beta_v[k] = r.beta;
        nu_v[k] = r.nu;
        l_det_v[k] = r.l_det;
    }

    // Normalize pi — sequential sum then divide, matching C order
    let d_np: f64 = pi_v.iter().sum();
    pi_v.iter_mut().for_each(|p| *p /= d_np);

    PerformMStepResult {
        mu: all_mu, m: all_m, covar: all_covar, sigma: all_sigma,
        pi: pi_v, beta: beta_v, nu: nu_v, l_det: l_det_v,
    }
}

const NOT_SET: i32 = -1;

/// K-means initialisation for the VBGMM.
///
/// Random initial assignment, then iterative k-means until convergence
/// or max_iter. Converts final hard assignments to soft Z and runs
/// performMStepMP. Uses the GSL RNG for reproducibility.
///
/// Returns (z, mstep_result) where z is the soft responsibility matrix.
///
/// c-concoct/c_vbgmm_fit.c:753-822
pub fn init_kmeans(
    n_samples: usize,
    n_dims: usize,
    n_clusters: usize,
    data: &[f64],
    seed: u64,
    max_iter: usize,
    vb_params: &VBParams,
) -> (Vec<f64>, PerformMStepResult) {
    use crate::c_ffi;

    let nn = n_samples;
    let nk = n_clusters;
    let nd = n_dims;

    let rng = unsafe { c_ffi::gsl_rng_new(seed) };

    // Random initial assignment
    let mut max_z = vec![0i32; nn];
    let mut weights = vec![0i32; nk];
    for i in 0..nn {
        let nik = unsafe { c_ffi::gsl_rng_uniform_int(rng, nk as u64) } as i32;
        max_z[i] = nik;
        weights[nik as usize] += 1;
    }

    // Initial means
    let mut mu_flat = vec![0.0f64; nk * nd];
    update_means(data, nn, nk, nd, &max_z, &weights, &mut mu_flat);

    // K-means iterations
    let mut n_change = nn;
    let mut n_iter = 0;

    while n_change > 0 && n_iter < max_iter {
        n_change = 0;

        for i in 0..nn {
            let data_row = &data[i * nd..(i + 1) * nd];
            let mut d_min_dist = f64::MAX;
            let mut n_min_k = NOT_SET;

            for k in 0..nk {
                let mu_row = &mu_flat[k * nd..(k + 1) * nd];
                let d_dist = calc_dist(data_row, mu_row);
                if d_dist < d_min_dist {
                    n_min_k = k as i32;
                    d_min_dist = d_dist;
                }
            }

            if n_min_k != max_z[i] {
                let n_curr = max_z[i];
                n_change += 1;
                weights[n_curr as usize] -= 1;
                weights[n_min_k as usize] += 1;
                max_z[i] = n_min_k;

                // Handle empty clusters
                if weights[n_curr as usize] == 0 {
                    let mut n_rand_i = unsafe {
                        c_ffi::gsl_rng_uniform_int(rng, nn as u64)
                    } as usize;

                    while weights[max_z[n_rand_i] as usize] == 1 {
                        n_rand_i = unsafe {
                            c_ffi::gsl_rng_uniform_int(rng, nn as u64)
                        } as usize;
                    }

                    let n_ki = max_z[n_rand_i];
                    weights[n_ki as usize] -= 1;
                    weights[n_curr as usize] = 1;
                    max_z[n_rand_i] = n_curr;
                }
            }
        }

        n_iter += 1;
        update_means(data, nn, nk, nd, &max_z, &weights, &mut mu_flat);
    }

    unsafe { c_ffi::gsl_rng_free(rng); }

    // Convert hard assignments to soft Z (1-hot)
    let mut z = vec![0.0f64; nn * nk];
    for i in 0..nn {
        z[i * nk + max_z[i] as usize] = 1.0;
    }

    // Run M-step
    let mstep_result = perform_mstep(nn, nd, nk, &z, data, vb_params);

    (z, mstep_result)
}

/// EM/VB training result.
pub struct TrainResult {
    /// Soft responsibilities, `[n_samples][n_clusters]`.
    pub z: Vec<f64>,
    /// Hard cluster assignments, length `n_samples`.
    pub assignments: Vec<i32>,
    /// Final variational lower bound.
    pub vbl: f64,
}

/// EM/VB training loop (Bishop Chapter 10).
///
/// Iterates: M-step → E-step → VBL until convergence (delta < epsilon)
/// or max_iter. Returns soft responsibilities, hard assignments, and
/// the final VBL.
///
/// c-concoct/c_vbgmm_fit.c:1048-1114
pub fn gmm_train_vb(
    n_samples: usize,
    n_dims: usize,
    n_clusters: usize,
    data: &[f64],
    z: &mut Vec<f64>,
    mstep_state: &mut PerformMStepResult,
    vb_params: &VBParams,
    log_wishart_b: f64,
    max_iter: usize,
    epsilon: f64,
) -> TrainResult {
    let nn = n_samples;
    let nk = n_clusters;
    let nd = n_dims;

    // Initial E-step + VBL
    calc_z(nn, nd, nk, data, z, &mstep_state.m, &mstep_state.sigma,
           &mstep_state.pi, &mstep_state.nu, &mstep_state.l_det, &mstep_state.beta);

    let mut vbl = calc_vbl(nn, nd, nk, z, &mstep_state.mu, &mstep_state.m,
                           &mstep_state.covar, &mut mstep_state.sigma,
                           &mstep_state.pi, &mstep_state.beta, &mstep_state.nu,
                           &mstep_state.l_det, &vb_params.inv_w0,
                           vb_params.beta0, vb_params.nu0, log_wishart_b);

    let mut n_iter = 0;
    let mut delta = f64::MAX;

    while n_iter < max_iter && delta > epsilon {
        // M-step
        *mstep_state = perform_mstep(nn, nd, nk, z, data, vb_params);

        // E-step
        calc_z(nn, nd, nk, data, z, &mstep_state.m, &mstep_state.sigma,
               &mstep_state.pi, &mstep_state.nu, &mstep_state.l_det, &mstep_state.beta);

        // VBL
        let last_vbl = vbl;
        vbl = calc_vbl(nn, nd, nk, z, &mstep_state.mu, &mstep_state.m,
                       &mstep_state.covar, &mut mstep_state.sigma,
                       &mstep_state.pi, &mstep_state.beta, &mstep_state.nu,
                       &mstep_state.l_det, &vb_params.inv_w0,
                       vb_params.beta0, vb_params.nu0, log_wishart_b);
        delta = (vbl - last_vbl).abs();

        // C prints to stderr here — we skip it to keep the Rust side
        // free of I/O overhead in benchmarks. The convergence values
        // are verified by the proptest instead.
        n_iter += 1;
    }

    // Hard assignments: argmax over Z per sample
    let mut assignments = vec![0i32; nn];
    for i in 0..nn {
        let z_row = &z[i * nk..(i + 1) * nk];
        let mut max_z_val = z_row[0];
        let mut max_k = 0i32;
        for k in 1..nk {
            if z_row[k] > max_z_val {
                max_k = k as i32;
                max_z_val = z_row[k];
            }
        }
        assignments[i] = max_k;
    }

    TrainResult {
        z: z.clone(),
        assignments,
        vbl,
    }
}

/// Remove empty clusters (pi=0) from the responsibility matrix and
/// re-index. Returns (new_z, new_assignments, new_n_clusters).
///
/// c-concoct/c_vbgmm_fit.c:429-505
pub fn compress_cluster(
    z: &[f64],
    pi: &[f64],
    n_samples: usize,
    n_clusters: usize,
) -> (Vec<f64>, Vec<i32>, usize) {
    let nn = n_samples;
    let nk = n_clusters;
    let dn = nn as f64;

    // Count active clusters
    let active: Vec<usize> = (0..nk).filter(|&k| pi[k] > 0.0).collect();
    let new_k = active.len();

    // Build compressed Z
    let mut new_z = vec![0.0f64; nn * new_k];
    for i in 0..nn {
        let z_row = &z[i * nk..(i + 1) * nk];
        let new_row = &mut new_z[i * new_k..(i + 1) * new_k];
        for (nc, &k) in active.iter().enumerate() {
            new_row[nc] = z_row[k];
        }
    }

    // Recalculate pi
    let mut new_pi = vec![0.0f64; new_k];
    for k in 0..new_k {
        for i in 0..nn {
            new_pi[k] += new_z[i * new_k + k];
        }
        new_pi[k] /= dn;
    }

    // Hard assignments: argmax per sample
    let mut assignments = vec![0i32; nn];
    for i in 0..nn {
        let row = &new_z[i * new_k..(i + 1) * new_k];
        let mut max_val = row[0];
        let mut max_k = 0i32;
        for k in 1..new_k {
            if row[k] > max_val {
                max_k = k as i32;
                max_val = row[k];
            }
        }
        assignments[i] = max_k;
    }

    (new_z, assignments, new_k)
}

/// Default VB parameters.
pub const DEF_BETA0: f64 = 1.0e-3;
pub const DEF_EPSILON: f64 = 1.0e-4;
pub const DEF_MAX_ITER: usize = 1000;

/// Full VBGMM fit — the public API matching c_vbgmm_fit().
///
/// Takes a flat row-major data matrix and returns cluster assignments.
/// This is the function that replaces the C extension.
///
/// c-concoct/c_vbgmm_fit.c:37-49 (c_vbgmm_fit) + 51-155 (driverMP)
pub fn vbgmm_fit(
    data: &[f64],
    n_samples: usize,
    n_dims: usize,
    n_clusters: usize,
    seed: u64,
    max_iter: usize,
) -> Vec<i32> {
    let nn = n_samples;
    let nd = n_dims;

    // Set up VB params (matches setVBParams)
    let beta0 = DEF_BETA0;
    let nu0 = nd as f64;
    let (var, _) = calc_sample_var(data, nn, nd);
    let mut inv_w0 = vec![0.0f64; nd * nd];
    for i in 0..nd {
        inv_w0[i * nd + i] = var[i] * (nd as f64);
    }
    let log_wishart_b = d_log_wishart_b(&inv_w0, nd, nu0, true);
    let vb_params = VBParams { beta0, nu0, inv_w0 };

    // Init + train
    let (mut z, mut mstep_state) = init_kmeans(
        nn, nd, n_clusters, data, seed, max_iter, &vb_params,
    );
    let result = gmm_train_vb(
        nn, nd, n_clusters, data, &mut z, &mut mstep_state,
        &vb_params, log_wishart_b, max_iter, DEF_EPSILON,
    );

    // Compress (remove empty clusters) and return assignments
    let (_new_z, assignments, _new_k) = compress_cluster(
        &result.z, &mstep_state.pi, nn, n_clusters,
    );

    assignments
}
