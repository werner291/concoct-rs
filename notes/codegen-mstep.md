# mstep performance analysis

## Summary

At 512 samples × 16 clusters × 32 dims, Rust mstep is ~9% slower than C
(99.7µs vs 91.2µs). At 128x8x16 Rust is 6% faster.

## Root cause: LLVM x86 loop vectorizer

Same issue as calcDist (see codegen-calcdist.md). GCC vectorizes independent
per-element operations (mul, sub) with packed SSE2 while keeping dependent
accumulations scalar. LLVM does not.

We manually applied SSE2 intrinsics (`_mm_mul_pd`, `_mm_add_pd`) to the
covariance and mean accumulation loops, matching GCC's codegen. This reduced
the 512x gap from 2.3x to 1.09x.

## Breakdown (C, 512x32)

- Weighted mean accumulation: ~5µs
- Covariance accumulation: ~79µs (dominates)
- Matrix building + symmetrise + Cholesky: ~7µs
- Total: ~91µs

## Per-section profiling (512 samples, 32 dims)

| Section | C | Rust | Diff |
|---------|---|------|------|
| Weighted mean | 5.6µs | 4.1µs | Rust 1.5µs faster (SSE2) |
| Covariance | 72.1µs | 73.6µs | Rust 1.5µs slower (SSE2 tail overhead) |
| Symmetrise + covar_out | 0.9µs | 1.4µs | Rust 0.5µs slower (scalar) |
| Eq 10.62 (InvWK) | 0.1µs | 0.5µs | Rust 0.4µs slower (scalar) |
| Sigma build | 0.7µs | 0.8µs | Rust 0.1µs slower (scalar) |
| Vec allocations | 0 | 0.4µs | C uses stack VLA + 2 mallocs |
| **Sections total** | **~79µs** | **~81µs** | **~2µs** |
| **Full mstep** | **~91µs** | **~100µs** | **~9µs** |

The ~7µs missing from sections vs full is Cholesky (GSL, same both sides)
plus loop control/branching overhead.

No single smoking gun. The gap is spread across every small loop where GCC
uses packed SSE2 and Rust stays scalar. Biggest opportunities: pre-allocated
buffers (-0.4µs) and SSE2 on sym+copy and eq1062 (-0.9µs).

## BLAS is not a win at these sizes

| Approach | 512×32 |
|----------|--------|
| Rust SSE2 | 99.7µs |
| C hand-written | 91.2µs |
| GSL cblas_dsyr × N | 170.7µs |
| GSL cblas_dgemm | 290.1µs |

BLAS per-call overhead dominates at D=32. The z > MIN_Z filter (skipping
~30% of samples) also can't be expressed in a single BLAS call.

## Future optimisation paths (post-translation)

1. Apply SSE2 intrinsics to the remaining scalar loops (Eq 10.62, sigma)
2. Reduce allocations: take pre-allocated buffers like updateMeans does
3. At larger dims (D≥64), BLAS dsyrk becomes competitive
4. With ndarray/nalgebra: express as matrix ops, let the BLAS backend handle it
5. If LLVM adds ordered-reduction vectorization for x86, all of this goes away
