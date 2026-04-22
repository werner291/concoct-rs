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

## Idiomatic Rust is bit-identical

Replacing index loops and SSE2 intrinsics with `chunks_exact`, `zip`,
`iter_mut().for_each()` produces the same float results — tested over 2,000
adversarial proptest cases with zero divergences. Rust's iterator chains are
sequential by specification, so the accumulation order is preserved.

However, LLVM generates purely scalar code for the iterator version regardless
of target CPU (tested baseline, x86-64-v3, znver4). The auto-vectorizer does
not fire. The performance gap is 2x at 512x16x32 compared to the hand-written
SSE2 intrinsics.

Decision: use the idiomatic version as the real implementation. It's safe,
readable, bit-identical, and the pipeline-level performance difference is
negligible (14ms over a full training run at worst). The SSE2 investigation
is preserved in git history as documentation.

## target-cpu experiments

All configurations maintain bit-exact equivalence with C (LLVM does not
emit FMA unless explicitly asked via `-C llvm-args=-fp-contract=fast`).

| Config | 512x16x32 Rust (SSE2) | 512x16x32 Rust (idiomatic) | C |
|--------|----------------------|---------------------------|---|
| x86-64 (baseline) | 110��s | 181µs | 91µs |
| x86-64-v3 (AVX2, no FMA) | 110µs | 181µs | 91µs |
| znver4 (native) | 87µs | 165µs | 90µs |
| znver4 + fp-contract=fast | 87µs | 165µs | 93µs |

Wider vectors don't help because: (a) our SSE2 intrinsics explicitly use
128-bit ops, (b) LLVM won't auto-vectorize the idiomatic loops regardless
of available instructions.

## BLAS is not competitive at these sizes

| Approach | 512×32 |
|----------|--------|
| Hand-written (C or Rust SSE2) | ~90µs |
| GSL cblas_dsyr × N | 171µs |
| GSL cblas_dgemm | 290µs |

Per-call overhead and inability to express the z > MIN_Z filter.

## Summary

The performance gap is LLVM's x86 codegen, not Rust. Hand-writing the same
SSE2 that GCC auto-generates closes the gap. But the idiomatic version is
correct, safe, and the end-to-end impact is small. Ship clean code.
