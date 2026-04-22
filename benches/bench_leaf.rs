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

fn make_matrix(n_samples: usize, n_dims: usize) -> Vec<f64> {
    (0..n_samples * n_dims)
        .map(|i| (i as f64) * 0.01 - (n_samples as f64) * (n_dims as f64) * 0.005)
        .collect()
}

fn bench_calc_sample_var(c: &mut Criterion) {
    let mut group = c.benchmark_group("calc_sample_var");
    for (n, d) in [(16, 8), (128, 32), (512, 64)] {
        let data = make_matrix(n, d);

        group.bench_with_input(BenchmarkId::new("rust", format!("{n}x{d}")), &(n, d), |b, _| {
            b.iter(|| vbgmm::calc_sample_var(black_box(&data), black_box(n), black_box(d)))
        });
        group.bench_with_input(BenchmarkId::new("c", format!("{n}x{d}")), &(n, d), |b, _| {
            let row_ptrs = c_ffi::RowPointers::new(&data, n, d);
            let cdata = row_ptrs.as_cdata(n, d);
            let mut var = vec![0.0f64; d];
            let mut mu = vec![0.0f64; d];
            b.iter(|| unsafe {
                c_ffi::calcSampleVar(black_box(&cdata), var.as_mut_ptr(), mu.as_mut_ptr())
            })
        });
    }
    group.finish();
}

fn bench_update_means(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_means");
    for (n, k, d) in [(64, 8, 16), (256, 16, 32), (1024, 32, 64)] {
        let data = make_matrix(n, d);
        let assignments: Vec<i32> = (0..n).map(|i| (i % k) as i32).collect();
        let mut weights = vec![0i32; k];
        for &a in &assignments { weights[a as usize] += 1; }

        let label = format!("{n}x{k}x{d}");

        group.bench_with_input(BenchmarkId::new("rust", &label), &(), |b, _| {
            let mut mu = vec![0.0f64; k * d];
            b.iter(|| vbgmm::update_means(
                black_box(&data), black_box(n), black_box(k), black_box(d),
                black_box(&assignments), black_box(&weights), black_box(&mut mu),
            ))
        });
        group.bench_with_input(BenchmarkId::new("c", &label), &(), |b, _| {
            let data_ptrs: Vec<*const f64> = (0..n)
                .map(|i| data[i * d..].as_ptr())
                .collect();
            let mut c_mu_flat = vec![0.0f64; k * d];
            let c_mu_ptrs: Vec<*mut f64> = (0..k)
                .map(|ki| c_mu_flat[ki * d..].as_mut_ptr())
                .collect();
            b.iter(|| unsafe {
                c_ffi::ffi_updateMeans(
                    black_box(data_ptrs.as_ptr()),
                    black_box(n as i32), black_box(k as i32), black_box(d as i32),
                    black_box(assignments.as_ptr()), black_box(weights.as_ptr()),
                    black_box(c_mu_ptrs.as_ptr()),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_calc_dist, bench_calc_sample_var, bench_update_means);
criterion_main!(benches);
