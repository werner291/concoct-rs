mod common;

use concoct::c_ffi;
use concoct::vbgmm;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10_000))]

    #[test]
    fn calc_dist_matches_c(
        dim in 1usize..64,
        seed_x in prop::collection::vec(common::adversarial_f64(), 64),
        seed_mu in prop::collection::vec(common::adversarial_f64(), 64),
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
