use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

/// Strategy that mixes adversarial floats with uniform random values.
/// Excludes NaN and infinity — calcDist does not document handling these.
fn adversarial_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0f64),
        Just(-0.0f64),
        Just(f64::MIN_POSITIVE),     // smallest positive normal
        Just(5e-324f64),             // smallest subnormal
        Just(f64::EPSILON),
        Just(f64::MAX),
        Just(f64::MIN),
        (-1e-300f64..1e-300),        // near-zero
        (-1e6f64..1e6),              // general range
        (-1e300f64..-1e290),         // large negative
        (1e290f64..1e300),           // large positive
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn calc_dist_matches_c(
        dim in 1usize..64,
        seed_x in prop::collection::vec(adversarial_f64(), 64),
        seed_mu in prop::collection::vec(adversarial_f64(), 64),
    ) {
        let x = &seed_x[..dim];
        let mu = &seed_mu[..dim];

        let rust_result = vbgmm::calc_dist(x, mu);
        let c_result = unsafe {
            c_ffi::calcDist(x.as_ptr(), mu.as_ptr(), dim as i32)
        };

        assert_eq!(
            rust_result.to_bits(),
            c_result.to_bits(),
            "dim={dim}: rust={rust_result:e} ({:016x}) vs c={c_result:e} ({:016x})",
            rust_result.to_bits(),
            c_result.to_bits(),
        );
    }
}
