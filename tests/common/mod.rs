use proptest::prelude::*;

/// Strategy that mixes adversarial floats with uniform random values.
/// Excludes NaN/Inf — the C functions do not document handling these.
pub fn adversarial_f64() -> impl Strategy<Value = f64> {
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
        Just(1.0f64),
        Just(-1.0f64),
    ]
}
