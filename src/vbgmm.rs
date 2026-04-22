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
