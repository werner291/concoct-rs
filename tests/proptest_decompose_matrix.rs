mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

/// Generate a symmetric positive-definite matrix by computing A^T * A + eps*I.
fn spd_matrix(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-10.0f64..10.0, n * n).prop_map(move |a| {
        let mut m = vec![0.0f64; n * n];
        // m = A^T * A
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += a[k * n + i] * a[k * n + j];
                }
                m[i * n + j] = sum;
            }
        }
        // + eps*I for numerical stability
        for i in 0..n {
            m[i * n + i] += 1.0;
        }
        m
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5_000))]

    #[test]
    fn decompose_matrix_matches_c(
        n in 2usize..16,
        matrix in spd_matrix(16),
    ) {
        let matrix: Vec<f64> = matrix.iter().copied().take(n * n).collect();

        // We need to rebuild a proper SPD matrix at size n (the strategy makes 16x16)
        let mut a = vec![0.0f64; n * n];
        // Use the first n*n elements as a source to build A^T*A + I
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += matrix[k * n + i] * matrix[k * n + j];
                }
                a[i * n + j] = sum;
            }
        }
        for i in 0..n {
            a[i * n + i] += 1.0;
        }

        // Rust copy
        let mut rust_matrix = a.clone();
        let rust_det = vbgmm::decompose_matrix(&mut rust_matrix, n);

        // C copy
        let mut c_matrix = a.clone();
        let c_det = unsafe {
            let gsl_m = c_ffi::gsl_matrix_from_flat(&c_matrix, n);
            let det = c_ffi::decomposeMatrix(gsl_m, n as i32);
            let result = c_ffi::gsl_matrix_to_flat(gsl_m, n);
            c_ffi::gsl_matrix_free(gsl_m);
            c_matrix = result;
            det
        };

        // Check log-determinant
        assert_eq!(
            rust_det.to_bits(), c_det.to_bits(),
            "det mismatch: rust={rust_det:e} vs c={c_det:e}",
        );

        // Check inverted matrix
        for i in 0..n {
            for j in 0..n {
                let idx = i * n + j;
                assert_eq!(
                    rust_matrix[idx].to_bits(), c_matrix[idx].to_bits(),
                    "matrix mismatch at [{i},{j}]: rust={:e} vs c={:e}",
                    rust_matrix[idx], c_matrix[idx],
                );
            }
        }
    }
}
