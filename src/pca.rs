// PCA via LAPACK dgesdd (divide-and-conquer SVD).
//
// Replaces sklearn.decomposition.PCA as used by concoct/transform.py.
// Calls the same LAPACK implementation (OpenBLAS via nix) that scipy uses,
// but without the Python/sklearn/scipy wrapper chain.
//
// The algorithm matches sklearn's PCA._fit_full with svd_flip(u_based_decision=False):
//   1. Center data (subtract column means)
//   2. Thin SVD via dgesdd
//   3. Sign-flip Vt rows so largest-abs element per row is positive
//   4. Select components by cumulative explained variance ratio
//   5. Project: (X - mean) @ components.T

use std::os::raw::c_char;

extern "C" {
    /// LAPACK divide-and-conquer SVD.
    /// Computes A = U * S * Vt where A is m×n (column-major).
    fn dgesdd_(
        jobz: *const c_char,
        m: *const i32,
        n: *const i32,
        a: *mut f64,
        lda: *const i32,
        s: *mut f64,
        u: *mut f64,
        ldu: *const i32,
        vt: *mut f64,
        ldvt: *const i32,
        work: *mut f64,
        lwork: *const i32,
        iwork: *mut i32,
        info: *mut i32,
    );
}

pub struct PcaResult {
    /// Transformed data, flat row-major (n_samples × n_components).
    pub transformed: Vec<f64>,
    /// PCA components, flat row-major (n_components × n_features).
    pub components: Vec<f64>,
    /// Number of components selected.
    pub n_components: usize,
    /// Number of input features.
    pub n_features: usize,
    /// Column means used for centering (length n_features).
    pub mean: Vec<f64>,
    /// Explained variance ratio for each selected component.
    pub explained_variance_ratio: Vec<f64>,
}

/// Perform PCA on a data matrix.
///
/// # Arguments
/// * `data` — flat row-major matrix, n_samples × n_features
/// * `n_samples` — number of rows
/// * `n_features` — number of columns
/// * `n_components` — if < 1.0: minimum cumulative variance ratio to retain;
///                    if >= 1.0: exact number of components (truncated to usize)
///
/// # Panics
/// If LAPACK dgesdd fails (info != 0).
pub fn pca(
    data: &[f64],
    n_samples: usize,
    n_features: usize,
    n_components: f64,
) -> PcaResult {
    assert_eq!(data.len(), n_samples * n_features);
    let k = n_samples.min(n_features);

    // 1. Column means
    let mut mean = vec![0.0f64; n_features];
    for row in data.chunks_exact(n_features) {
        for (j, &val) in row.iter().enumerate() {
            mean[j] += val;
        }
    }
    let n_f64 = n_samples as f64;
    for m in mean.iter_mut() {
        *m /= n_f64;
    }

    // 2. Center data (make a copy)
    let mut centered = data.to_vec();
    for row in centered.chunks_exact_mut(n_features) {
        for (j, val) in row.iter_mut().enumerate() {
            *val -= mean[j];
        }
    }

    // 3. SVD via LAPACK dgesdd
    //
    // Our data is row-major (n_samples × n_features). When interpreted as
    // column-major (Fortran order), LAPACK sees the transpose: an
    // n_features × n_samples matrix.
    //
    // dgesdd(A^T) = U' * S * Vt' where A = (Vt')^T * S * (U')^T
    //
    // Reading LAPACK's column-major outputs as row-major:
    //   LAPACK U'  (col-major n_features × K) → row-major K × n_features = our Vt
    //   LAPACK Vt' (col-major K × n_samples)  → row-major n_samples × K  = our U
    let m_lap = n_features as i32;
    let n_lap = n_samples as i32;
    let k_i32 = k as i32;

    let mut s = vec![0.0f64; k];
    let mut u_lap = vec![0.0f64; n_features * k]; // LAPACK's U → our Vt
    let mut vt_lap = vec![0.0f64; k * n_samples]; // LAPACK's Vt → our U
    let mut info: i32 = 0;
    let mut iwork = vec![0i32; 8 * k];

    // Workspace query
    let jobz = b'S' as c_char;
    let mut work_query = [0.0f64; 1];
    let lwork_query: i32 = -1;
    unsafe {
        dgesdd_(
            &jobz,
            &m_lap, &n_lap,
            centered.as_mut_ptr(), &m_lap,
            s.as_mut_ptr(),
            u_lap.as_mut_ptr(), &m_lap,
            vt_lap.as_mut_ptr(), &k_i32,
            work_query.as_mut_ptr(), &lwork_query,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    assert_eq!(info, 0, "dgesdd workspace query failed: info={info}");

    let lwork = work_query[0] as i32;
    let mut work = vec![0.0f64; lwork as usize];

    // Actual SVD
    unsafe {
        dgesdd_(
            &jobz,
            &m_lap, &n_lap,
            centered.as_mut_ptr(), &m_lap,
            s.as_mut_ptr(),
            u_lap.as_mut_ptr(), &m_lap,
            vt_lap.as_mut_ptr(), &k_i32,
            work.as_mut_ptr(), &lwork,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    assert_eq!(info, 0, "dgesdd SVD failed: info={info}");

    // u_lap is now our Vt: K × n_features (row-major)
    let vt = u_lap;

    // 4. Sign correction (sklearn svd_flip with u_based_decision=False)
    //
    // For each row of Vt, find the column with the largest absolute value.
    // If that element is negative, flip the entire row's sign.
    let mut vt = vt;
    for i in 0..k {
        let row = &vt[i * n_features..(i + 1) * n_features];
        let max_col = row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if row[max_col] < 0.0 {
            for val in vt[i * n_features..(i + 1) * n_features].iter_mut() {
                *val = -*val;
            }
        }
    }

    // 5. Explained variance and component selection
    let explained_var: Vec<f64> = s.iter()
        .map(|&si| si * si / (n_samples as f64 - 1.0))
        .collect();
    let total_var: f64 = explained_var.iter().sum();
    let ratio: Vec<f64> = explained_var.iter()
        .map(|&ev| ev / total_var)
        .collect();

    let n_selected = if n_components >= 1.0 {
        // Exact number of components
        (n_components as usize).min(k)
    } else {
        // Select by cumulative variance ratio (matches sklearn's
        // searchsorted(cumsum, n_components, side='right') + 1)
        let mut cumsum = 0.0;
        let mut selected = k;
        for (i, &r) in ratio.iter().enumerate() {
            cumsum += r;
            if cumsum > n_components {
                selected = i + 1;
                break;
            }
        }
        selected
    };

    // 6. Extract components: Vt[:n_selected, :] → n_selected × n_features
    let components = vt[..n_selected * n_features].to_vec();
    let explained_variance_ratio = ratio[..n_selected].to_vec();

    // 7. Transform: X_centered @ components.T
    //    (n_samples × n_features) @ (n_features × n_selected) = n_samples × n_selected
    //
    // Re-center from original data (centered was destroyed by dgesdd)
    let mut transformed = vec![0.0f64; n_samples * n_selected];
    for i in 0..n_samples {
        for c in 0..n_selected {
            let mut dot = 0.0f64;
            for j in 0..n_features {
                dot += (data[i * n_features + j] - mean[j]) * components[c * n_features + j];
            }
            transformed[i * n_selected + c] = dot;
        }
    }

    PcaResult {
        transformed,
        components,
        n_components: n_selected,
        n_features,
        mean,
        explained_variance_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_basic_properties() {
        // Simple 4×3 matrix with clear principal direction
        let data = [
            1.0, 0.0, 0.0,
            2.0, 0.1, 0.0,
            3.0, 0.0, 0.1,
            4.0, 0.1, 0.1,
        ];
        let result = pca(&data, 4, 3, 0.9);

        // Should select at least 1 component (first axis has most variance)
        assert!(result.n_components >= 1);
        assert!(result.n_components <= 3);

        // Transformed data should have correct dimensions
        assert_eq!(result.transformed.len(), 4 * result.n_components);
        assert_eq!(result.components.len(), result.n_components * 3);

        // Explained variance ratio should sum to <= 1.0
        let sum: f64 = result.explained_variance_ratio.iter().sum();
        assert!(sum <= 1.0 + 1e-10);
        assert!(sum >= 0.9); // we asked for 90%

        // Mean should be [2.5, 0.05, 0.05]
        assert!((result.mean[0] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn pca_exact_components() {
        // Request exactly 2 components
        let data = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ];
        let result = pca(&data, 4, 3, 2.0);
        assert_eq!(result.n_components, 2);
        assert_eq!(result.transformed.len(), 4 * 2);
        assert_eq!(result.components.len(), 2 * 3);
    }
}
