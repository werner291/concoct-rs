use concoct::c_ffi;
use concoct::vbgmm;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_vecs(dim: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.1 - (dim as f64) * 0.05).collect();
    let mu: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.05 + 0.3).collect();
    (x, mu)
}

fn bench_calc_dist(c: &mut Criterion) {
    let mut group = c.benchmark_group("calc_dist");
    for dim in [4, 32, 128] {
        let (x, mu) = make_vecs(dim);
        let d = dim as i32;

        group.bench_with_input(BenchmarkId::new("rust", dim), &dim, |b, _| {
            b.iter(|| vbgmm::calc_dist(black_box(&x), black_box(&mu)))
        });
        group.bench_with_input(BenchmarkId::new("c", dim), &dim, |b, _| {
            b.iter(|| unsafe { c_ffi::calcDist(black_box(x.as_ptr()), black_box(mu.as_ptr()), black_box(d)) })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_calc_dist);
criterion_main!(benches);
