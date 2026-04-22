mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

/// Generate a symmetric positive-definite matrix (flat, n x n).
fn spd_matrix(n: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-10.0f64..10.0, n * n).prop_map(move |a| {
        let mut m = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += a[k * n + i] * a[k * n + j];
                }
                m[i * n + j] = sum;
            }
        }
        for i in 0..n {
            m[i * n + i] += 1.0;
        }
        m
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5_000))]

    #[test]
    fn d_log_wishart_b_matches_c(
        n in 2usize..12,
        source in prop::collection::vec(-10.0f64..10.0, 12 * 12),
        nu_offset in 0.0f64..10.0,
        b_inv in prop::bool::ANY,
    ) {
        // Build SPD matrix at size n
        let mut matrix = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += source[k * n + i] * source[k * n + j];
                }
                matrix[i * n + j] = sum;
            }
        }
        for i in 0..n {
            matrix[i * n + i] += 1.0;
        }

        let nu = (n as f64) + nu_offset; // nu must be >= nD

        let rust_result = vbgmm::d_log_wishart_b(&matrix, n, nu, b_inv);

        let c_result = unsafe {
            let gsl_m = c_ffi::gsl_matrix_from_flat(&matrix, n);
            let r = c_ffi::dLogWishartB(gsl_m, n as i32, nu, if b_inv { 1 } else { 0 });
            c_ffi::gsl_matrix_free(gsl_m);
            r
        };

        assert_eq!(
            rust_result.to_bits(), c_result.to_bits(),
            "n={n} nu={nu} b_inv={b_inv}: rust={rust_result:e} vs c={c_result:e}",
        );
    }

    #[test]
    fn d_wishart_expect_log_det_matches_c(
        n in 2usize..12,
        source in prop::collection::vec(-10.0f64..10.0, 12 * 12),
        nu_offset in 0.0f64..10.0,
    ) {
        let mut matrix = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += source[k * n + i] * source[k * n + j];
                }
                matrix[i * n + j] = sum;
            }
        }
        for i in 0..n {
            matrix[i * n + i] += 1.0;
        }

        let nu = (n as f64) + nu_offset;

        let mut rust_matrix = matrix.clone();
        let rust_result = vbgmm::d_wishart_expect_log_det(&mut rust_matrix, n, nu);

        let c_result = unsafe {
            let gsl_m = c_ffi::gsl_matrix_from_flat(&matrix, n);
            let r = c_ffi::dWishartExpectLogDet(gsl_m, nu, n as i32);
            let c_matrix_after = c_ffi::gsl_matrix_to_flat(gsl_m, n);
            c_ffi::gsl_matrix_free(gsl_m);

            // Also check the in-place modified matrix matches
            for i in 0..n {
                for j in 0..n {
                    let idx = i * n + j;
                    assert_eq!(
                        rust_matrix[idx].to_bits(), c_matrix_after[idx].to_bits(),
                        "matrix mismatch at [{i},{j}]",
                    );
                }
            }
            r
        };

        assert_eq!(
            rust_result.to_bits(), c_result.to_bits(),
            "n={n} nu={nu}: rust={rust_result:e} vs c={c_result:e}",
        );
    }
}
