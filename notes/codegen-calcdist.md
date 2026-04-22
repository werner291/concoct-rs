# calcDist codegen: GCC vs LLVM on x86-64 baseline

Reproduce:

    nix build .#packages.x86_64-linux.codegen-calcdist
    cat result/calcdist_c_func.s    # GCC output
    cat result/calcdist_rust_func.s # LLVM output

## Setup

Both compilers target x86-64 baseline (SSE2, no AVX). GCC 15.2 with `-O3
-std=c99`, rustc/LLVM with `-C opt-level=3 -C target-cpu=x86-64`.

## GCC (C): packed sub/mul, scalar accumulation

```asm
.L4:
    movupd  (%rdi,%rax), %xmm1       ; load 2 doubles from x
    movupd  (%rsi,%rax), %xmm3       ; load 2 doubles from mu
    subpd   %xmm3, %xmm1            ; packed subtract (2 at once)
    mulpd   %xmm1, %xmm1            ; packed square (2 at once)
    addsd   %xmm1, %xmm0            ; accumulate low element
    unpckhpd %xmm1, %xmm1           ; move high to low
    addsd   %xmm1, %xmm0            ; accumulate high element
```

GCC vectorizes the independent per-element operations (sub, mul) with packed
SSE2 but keeps the dependent accumulation (add) scalar. This preserves the
left-to-right accumulation order while halving the sub/mul instruction count.

## LLVM (Rust): scalar everything, unrolled x4

```asm
.LBB0_10:
    movsd   (%rdi,%rcx,8), %xmm1    ; load 1 double
    subsd   (%rdx,%rcx,8), %xmm1    ; scalar subtract
    mulsd   %xmm1, %xmm1            ; scalar square
    addsd   %xmm0, %xmm1            ; accumulate
    ; ... repeat 3 more times
```

LLVM unrolls by 4 but does not vectorize the sub/mul. Every operation is scalar.

## Benchmark (ns)

| dim | Rust | C | ratio |
|-----|------|---|-------|
| 4   | 2.34 | 2.63 | 0.89x (Rust faster) |
| 32  | 17.2 | 15.5 | 1.11x |
| 128 | 82.1 | 77.5 | 1.06x |

At dim=4 Rust wins on loop overhead. At larger dims, GCC's packed sub/mul wins.

## `#[inline(never)]` is redundant for this function

Removing it does not change benchmarks or break bit-exact equivalence (10,000
proptest cases). The `target-cpu=x86-64` pin is what prevents float divergence.
May not hold for more complex functions.

## This is a known-class LLVM missed optimisation on x86

LLVM has support for "in-order (strict) FP reductions" — vectorizing
independent operations while keeping the dependent accumulation sequential.
This is implemented for AArch64 and RISC-V but **not for x86**. GCC does
this on x86 already (the `subpd`/`mulpd` + scalar `addsd` pattern above).

See: https://llvm.org/docs/Vectorizers.html (search "ordered reductions")

This means the performance gap is not a fundamental Rust limitation — it's a
missing x86 backend feature in LLVM that already has the right infrastructure
on other architectures.

## Experiments

All experiments preserve bit-exact equivalence (10,000 proptest cases).

| Configuration | dim=4 | dim=32 | dim=128 |
|---|---|---|---|
| C (GCC -O3, baseline) | 2.63ns | 15.5ns | 77.5ns |
| Rust, `inline(never)`, `target-cpu=x86-64` | 2.34ns | 17.2ns | 82.1ns |
| Rust, `inline`, `target-cpu=x86-64` | 2.34ns | 17.2ns | 82.1ns |
| Rust, `inline(never)`, `target-cpu=native` | 3.10ns | 16.9ns | 82.6ns |
| Rust, `inline`, `target-cpu=native` | 2.42ns | 16.7ns | 83.5ns |

Findings:

- `#[inline(never)]` has no effect at `target-cpu=x86-64` (same numbers).
  The target-cpu pin is the actual guard against float divergence. Removed
  the inline ban for calcDist — it was redundant. Future functions should
  be tested the same way before deciding.
- `target-cpu=native` does not close the gap — LLVM still doesn't do the
  partial vectorization trick even with AVX available. The x86 loop
  vectorizer is all-or-nothing: since the reduction is ordered, it picks
  nothing.
- Full `opt-level=3 + target-cpu=native + #[inline]` produces the same
  numbers. The gap is structural to LLVM's x86 backend.
- Closing it requires explicit SIMD or an upstream LLVM fix.
- `fp-contract=fast` (allowing FMA) makes things *worse* at dim=128
  (98ns vs 82ns). LLVM makes bad FMA decisions for this pattern.
- Not a bottleneck — the training loop dominates runtime. The real
  performance wins will come from the composite functions where memory
  layout, allocation, and cache behaviour matter more than instruction
  selection.
