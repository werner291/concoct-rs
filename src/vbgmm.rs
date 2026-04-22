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
