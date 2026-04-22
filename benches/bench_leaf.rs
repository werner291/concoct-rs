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

fn bench_mstep(c: &mut Criterion) {
    let mut group = c.benchmark_group("mstep");
    for (n, k, d) in [(32, 4, 4), (128, 8, 16), (512, 16, 32)] {
        let data = make_matrix(n, d);
        // Uniform-ish responsibilities
        let z: Vec<f64> = (0..n * k).map(|i| {
            if i % k == 0 { 0.7 } else { 0.3 / (k as f64 - 1.0) }
        }).collect();
        // SPD prior
        let mut inv_w0 = vec![0.0f64; d * d];
        for i in 0..d { inv_w0[i * d + i] = 1.0; }

        let vb_params = vbgmm::VBParams {
            beta0: 0.001,
            nu0: d as f64,
            inv_w0: inv_w0.clone(),
        };

        let label = format!("{n}x{k}x{d}");

        group.bench_with_input(BenchmarkId::new("rust", &label), &(), |b, _| {
            b.iter(|| vbgmm::mstep(
                black_box(0), black_box(n), black_box(d), black_box(k),
                black_box(&z), black_box(&data), black_box(&vb_params),
            ))
        });
        group.bench_with_input(BenchmarkId::new("c", &label), &(), |b, _| {
            let data_ptrs: Vec<*const f64> = (0..n).map(|i| data[i * d..].as_ptr()).collect();
            let z_ptrs: Vec<*const f64> = (0..n).map(|i| z[i * k..].as_ptr()).collect();
            let inv_w0_ptrs: Vec<*const f64> = (0..d).map(|i| inv_w0[i * d..].as_ptr()).collect();

            let mut c_mu = vec![0.0f64; d];
            let mut c_m = vec![0.0f64; d];
            let mut c_pi = 0.0f64;
            let mut c_beta = 0.0f64;
            let mut c_nu = 0.0f64;
            let mut c_ldet = 0.0f64;
            let mut c_covar = vec![0.0f64; d * d];
            let mut c_sigma = vec![0.0f64; d * d];

            b.iter(|| unsafe {
                c_ffi::ffi_mstep(
                    black_box(0), black_box(n as i32), black_box(d as i32), black_box(k as i32),
                    black_box(z_ptrs.as_ptr()), black_box(data_ptrs.as_ptr()),
                    black_box(0.001), black_box(d as f64), black_box(inv_w0_ptrs.as_ptr()),
                    c_mu.as_mut_ptr(), c_m.as_mut_ptr(),
                    &mut c_pi, &mut c_beta, &mut c_nu, &mut c_ldet,
                    c_covar.as_mut_ptr(), c_sigma.as_mut_ptr(),
                )
            })
        });
    }
    group.finish();
}

fn bench_perform_mstep(c: &mut Criterion) {
    let mut group = c.benchmark_group("perform_mstep");
    for (n, k, d) in [(32, 4, 4), (128, 8, 16), (512, 16, 32)] {
        let data = make_matrix(n, d);
        let z: Vec<f64> = (0..n * k).map(|i| {
            if i % k == 0 { 0.7 } else { 0.3 / (k as f64 - 1.0) }
        }).collect();
        let mut inv_w0 = vec![0.0f64; d * d];
        for i in 0..d { inv_w0[i * d + i] = 1.0; }

        let vb_params = vbgmm::VBParams {
            beta0: 0.001,
            nu0: d as f64,
            inv_w0: inv_w0.clone(),
        };

        let label = format!("{n}x{k}x{d}");

        group.bench_with_input(BenchmarkId::new("rust", &label), &(), |b, _| {
            b.iter(|| vbgmm::perform_mstep(
                black_box(n), black_box(d), black_box(k),
                black_box(&z), black_box(&data), black_box(&vb_params),
            ))
        });
    }
    group.finish();
}

fn bench_train(c: &mut Criterion) {
    // Warm up both thread pools before measurement
    rayon::ThreadPoolBuilder::new().build_global().ok();
    // Dummy work to ensure Rayon pool is alive
    use rayon::prelude::*;
    let _: Vec<i32> = (0..100).into_par_iter().map(|x| x * 2).collect();
    let mut group = c.benchmark_group("train");
    for (n, k, d) in [(128, 8, 16), (512, 16, 32), (2048, 32, 64)] {
        let data = make_matrix(n, d);
        let beta0 = 0.001f64;
        let nu0 = d as f64;
        let (var, _) = vbgmm::calc_sample_var(&data, n, d);
        let mut inv_w0 = vec![0.0f64; d * d];
        for i in 0..d { inv_w0[i * d + i] = var[i] * (d as f64); }
        let log_wishart_b = vbgmm::d_log_wishart_b(&inv_w0, d, nu0, true);
        let vb_params = vbgmm::VBParams { beta0, nu0, inv_w0: inv_w0.clone() };
        let seed = 1u64;

        let label = format!("{n}x{k}x{d}");

        group.bench_with_input(BenchmarkId::new("rust", &label), &(), |b, _| {
            b.iter(|| {
                let (mut z, mut ms) = vbgmm::init_kmeans(
                    black_box(n), black_box(d), black_box(k),
                    black_box(&data), black_box(seed), black_box(1000),
                    black_box(&vb_params),
                );
                vbgmm::gmm_train_vb(
                    black_box(n), black_box(d), black_box(k),
                    black_box(&data), &mut z, &mut ms,
                    black_box(&vb_params), black_box(log_wishart_b),
                    black_box(1000), black_box(1.0e-4),
                )
            })
        });

        let data_ptrs: Vec<*const f64> = (0..n).map(|i| data[i * d..].as_ptr()).collect();
        group.bench_with_input(BenchmarkId::new("c", &label), &(), |b, _| {
            let mut assign = vec![0i32; n];
            b.iter(|| unsafe {
                c_ffi::ffi_trainFull(
                    black_box(data_ptrs.as_ptr()),
                    black_box(n as i32), black_box(k as i32), black_box(d as i32),
                    black_box(seed), black_box(1000), black_box(1.0e-4),
                    assign.as_mut_ptr(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_calc_dist, bench_calc_sample_var, bench_update_means, bench_mstep, bench_perform_mstep, bench_train);
criterion_main!(benches);
