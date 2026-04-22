use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

/// Strategy that mixes adversarial floats with uniform random values.
/// Excludes NaN/Inf — calcSampleVar does not document handling these.
fn adversarial_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0f64),
        Just(-0.0f64),
        Just(f64::MIN_POSITIVE),
        Just(5e-324f64),
        Just(f64::EPSILON),
        (-1e-300f64..1e-300),
        (-1e6f64..1e6),
        (-1e150f64..-1e140),
        (1e140f64..1e150),
        // Duplicate values (triggers zero variance per-dimension)
        Just(1.0f64),
        Just(-1.0f64),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn calc_sample_var_matches_c(
        n_samples in 2usize..128,
        n_dims in 1usize..32,
        seed in prop::collection::vec(adversarial_f64(), 128 * 32),
    ) {
        let data: Vec<f64> = seed.iter().copied().take(n_samples * n_dims).collect();
        prop_assume!(data.len() == n_samples * n_dims);

        let (rust_var, rust_mu) = vbgmm::calc_sample_var(&data, n_samples, n_dims);

        let row_ptrs = c_ffi::RowPointers::new(&data, n_samples, n_dims);
        let cdata = row_ptrs.as_cdata(n_samples, n_dims);
        let mut c_var = vec![0.0f64; n_dims];
        let mut c_mu = vec![0.0f64; n_dims];
        unsafe {
            c_ffi::calcSampleVar(&cdata, c_var.as_mut_ptr(), c_mu.as_mut_ptr());
        }

        for i in 0..n_dims {
            assert_eq!(
                rust_mu[i].to_bits(),
                c_mu[i].to_bits(),
                "mu mismatch at dim={i}, n_samples={n_samples}, n_dims={n_dims}: \
                 rust={:e} ({:016x}) vs c={:e} ({:016x})",
                rust_mu[i], rust_mu[i].to_bits(),
                c_mu[i], c_mu[i].to_bits(),
            );
            assert_eq!(
                rust_var[i].to_bits(),
                c_var[i].to_bits(),
                "var mismatch at dim={i}, n_samples={n_samples}, n_dims={n_dims}: \
                 rust={:e} ({:016x}) vs c={:e} ({:016x})",
                rust_var[i], rust_var[i].to_bits(),
                c_var[i], c_var[i].to_bits(),
            );
        }
    }
}
