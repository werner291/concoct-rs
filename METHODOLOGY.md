# Rewrite Methodology: concoct-rs

## Motivation

This project is a gradual rewrite of
[CONCOCT](https://github.com/BinPro/CONCOCT) (a Python+C metagenomic contig
binner) into Rust. An earlier project ([maxbin-rs](https://github.com/werner291/maxbin-rs)) did a
component-level rewrite successfully, but the maxbin-rs rewrite lacked a
disciplined commit-by-commit chain of provable changes. This made it harder to
trace *why* a particular decision was made and to verify correctness at each step
after the fact.

concoct-rs builds on that experience. The rewrite is still component-level and
inside-out, but this time we emphasise **traceability**: the Python original
stays runnable throughout the migration, and every commit carries a test that
proves the change is correct. The VCS history itself becomes documentation of the
rewrite — each commit establishes what changed, why, and how to verify it.

## Principles

### 1. One commit, one change, one proof

Every commit in this repository must contain:

1. **A change**: a single, well-scoped modification (a function ported, an
   interface introduced, a dependency replaced).
2. **A proof**: a test that exercises exactly that change and can be reproduced
   with a single nix command.

The commit message must explain *why* the change was made, what effect it has on
the overall system, and cite the test that proves it. For example:

```
Port calcSampleVar from C to Rust

Replace the C implementation of calcSampleVar (c-concoct/c_vbgmm_fit.c:202-227)
with an equivalent Rust function. This is the first leaf function in the VBGMM
call graph — it computes per-dimension sample mean and variance from the data
matrix, used by setVBParams to initialise the Wishart prior.

The Rust version preserves the exact accumulation order of the C original:
outer loop over dimensions, inner loop over samples, with the variance computed
as (sum_sq - N*mu^2)/(N-1). This order matters for float reproducibility.

Proof: nix build .#checks.x86_64-linux.test-calc-sample-var
```

### 2. Inside-out rewrite order

The CONCOCT pipeline is:

```
CLI args (parser.py)
  → load data (input.py)
    → PCA (transform.py)
      → VBGMM clustering (c_vbgmm_fit.c)  ← start here
    → write results (output.py)
```

We start at the bottom: the C extension (`c-concoct/c_vbgmm_fit.c`, ~1300 lines)
and the VBGMM core (`c-concoct/vbgmm.c`, ~12000 lines). These are the
performance-critical, correctness-critical, and hardest-to-debug components.

Within the C core, we further go inside-out. The VBGMM call graph is roughly:

```
c_vbgmm_fit
  → driverMP
    → generateInputData      (trivial reshape)
    → setVBParams             (calls calcSampleVar, dLogWishartB)
    → allocateCluster         (allocation)
    → fitEM_MP
      → initKMeans            (calls calcDist, updateMeans, performMStepMP)
      → gmmTrainVB_MP
        → calcZ_MP            (E-step: responsibilities)
        → calcVBL_MP          (variational lower bound; calls eqnA, eqnB)
        → performMStepMP      (M-step; calls mstep per component)
    → compressCluster         (post-processing)
    → calcCovarMatrices       (final covariance computation)
```

Leaf functions first: `calcDist`, `calcSampleVar`, `decomposeMatrix`,
`dLogWishartB`, `dWishartExpectLogDet`. Then composites: `mstep`, `calcZ_MP`,
`calcVBL_MP`. Then the training loop. Then the driver.

### 3. Bit-for-bit float equivalence

The Rust rewrite must produce **bit-identical** output to the C original for the
same inputs. Not "close enough". Not within epsilon. Identical.

This is non-trivial. Prior experience
([maxbin-rs](https://github.com/werner291/maxbin-rs)) identified three
categories of float divergence:

- **Order of operations**: any sequence of float operations — sums, products,
  divisions, compound expressions — must execute in the same order as the C
  code. This is not limited to addition;
  [maxbin-rs](https://github.com/werner291/maxbin-rs) was bitten by
  multiplication order specifically. The Rust code must mirror the C loop
  structure and operation order exactly.
- **Associativity**: chaining N float operations in a different order produces
  different results. If the C code iterates over samples in the inner loop,
  the Rust code must too.
- **Display length differences**: `printf("%f", x)` and Rust's `format!("{}", x)`
  produce different decimal representations. Output formatting must match the C
  `printf` format specifiers exactly.

Approach:

- Use `#[inline(never)]` and explicit parenthesisation where needed to prevent
  the compiler from reordering float arithmetic. These inline bans are
  scaffolding: they may be removed later, but only *after* equivalence is
  established, and the removal is its own commit with its own proof.
- Port functions one-to-one, preserving loop structure, variable naming (where
  it aids comparison), and accumulation order.
- Where the C code uses GSL (BLAS, Cholesky, digamma), identify the exact
  algorithm and reproduce it — or call the same GSL routines via FFI until the
  Rust equivalent is proven identical.

### 4. Two kinds of tests

#### a. Property-based equivalence tests (proptest)

For each ported function, we write a proptest that:

1. Generates **adversarial** inputs — not just uniform random values, but inputs
   specifically chosen to provoke float divergence. The input strategy must
   include:
   - **Special floats**: `±0.0`, subnormals, `±INF`, `NaN`, `MAX`, `MIN`,
     `MIN_POSITIVE`, `EPSILON`.
   - **Near-equal values**: pairs of floats that differ by 1 ULP, to stress
     subtraction cancellation.
   - **Mixed magnitudes**: combining very large and very small values in the same
     input to provoke catastrophic cancellation in sums and products.
   - **Exact duplicates**: repeated values in the input, which can trigger
     different codepaths (e.g., zero variance, singular covariance).
   - **Boundary sizes**: empty inputs, single-element inputs, and sizes at
     power-of-two boundaries.
2. Calls both the original C function (via FFI) and the new Rust function.
3. Asserts **bit-identical** output (not approximate equality).

This runs both implementations side by side in the same test binary. The nix
flake builds both the C library and the Rust crate, links them, and runs the
proptest suite.

Example test structure:

```rust
use proptest::prelude::*;

/// Strategy that mixes adversarial special floats with uniform random values.
fn adversarial_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        // Special values
        Just(0.0f64),
        Just(-0.0f64),
        Just(f64::MIN_POSITIVE),
        Just(f64::EPSILON),
        Just(f64::MAX),
        Just(f64::MIN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        Just(f64::NAN),
        // Subnormals
        (1u64..((1u64 << 52) - 1)).prop_map(|bits| f64::from_bits(bits)),
        // Near-zero
        (-1e-300f64..1e-300),
        // General range
        (-1e6f64..1e6),
        // Large magnitudes
        prop_oneof![(-1e300f64..-1e290), (1e290f64..1e300)],
    ]
}

proptest! {
    #[test]
    fn calc_sample_var_matches_c(
        data in vec(vec(adversarial_f64(), 1..64usize), 2..512usize),
    ) {
        let (rust_var, rust_mu) = rust::calc_sample_var(&data);
        let (c_var, c_mu) = c_ffi::calc_sample_var(&data);
        assert_eq!(rust_var.to_bits(), c_var.to_bits());
        assert_eq!(rust_mu.to_bits(), c_mu.to_bits());
    }
}
```

The bit comparison uses `f64::to_bits()` so that NaN, signed zero, and
subnormals are all caught.

Run with: `nix build .#checks.x86_64-linux.proptest-<function-name>`

#### b. Benchmark comparisons

For each ported function, we write a criterion benchmark that:

1. Runs the C function on a fixed dataset.
2. Runs the Rust function on the same dataset.
3. Reports wall-clock time and throughput.

Benchmarks are **informational by default**: they are logged and can be compared
across commits. However, a Rust function that is measurably *slower* than its C
equivalent is considered a failure and must be investigated before merging.

Run with: `nix build .#checks.x86_64-linux.bench-<function-name>`

### 5. Reproducibility through nix

Every test must be runnable via `nix flake check` or a specific
`nix build .#checks.<system>.<test-name>` invocation. No ambient state, no
"works on my machine". The flake pins:

- The exact nixpkgs revision (and therefore the exact GCC, GSL, Python, and
  Rust toolchain versions).
- The exact source tree (flake inputs are content-addressed).
- The exact test data (checked into the repository or fetched as a fixed-output
  derivation).

This means that anyone with nix can reproduce any test from any commit in the
history, which is essential for bisecting regressions.

### 6. Citation and traceability

Commit messages and test names must cite their sources:

- **C source location**: file and line range (e.g.,
  `c-concoct/c_vbgmm_fit.c:202-227`).
- **Algorithm reference**: Bishop equation numbers where applicable (the C code
  already references Bishop's PRML, e.g., "Bishop 10.60", "Equation 10.65").
- **Test command**: the exact `nix build` or `nix flake check` invocation that
  reproduces the proof.
- **Benchmark baseline**: the commit hash of the C benchmark baseline, if
  comparing performance.

## Rewrite phases

### Phase 0: Scaffolding (complete)

- [x] Nix flake with dev shell, Python package, and test infrastructure.
- [x] Migrate test suite from nose to pytest.
- [x] Hook existing tests into `nix flake check`.

### Phase 1: Leaf functions (complete)

Rust crate skeleton with C FFI oracle, then six leaf functions ported:
`calcDist`, `calcSampleVar`, `updateMeans` (pure Rust), `decomposeMatrix`,
`dLogWishartB`, `dWishartExpectLogDet` (GSL via FFI).

### Phase 2: Composite functions (complete)

Five composite functions ported: `mstep`, `calcZ_MP`, `calcVBL_MP`,
`performMStepMP`, `initKMeans`.

Key finding: idiomatic Rust (iterators, `chunks_exact`, `zip`) produces
bit-identical float results to C-style index loops. The performance gap
is LLVM's x86 loop vectorizer, not the language. Explicit AVX2 intrinsics
on the covariance inner loop beat GCC by 26% (see notes/codegen-mstep.md).

Functions that call GSL (`calcZ_MP`, `calcVBL_MP`, `decomposeMatrix`, etc.)
are benchmarked only for the Rust wrapper overhead — meaningful C-vs-Rust
benchmarks are deferred until GSL is replaced with Rust implementations.

### Phase 3: Training loop and driver (current)

- `gmmTrainVB_MP` (L1048-1114): The EM/VB iteration loop.
- `fitEM_MP` (L397-427): Top-level fit orchestration.
- `compressCluster` (L429-505): Post-fit cluster compression.
- `driverMP` (L51-155): The full driver.

At this point, the Rust VBGMM can replace the C extension entirely.

### Phase 4: Python integration layer (complete)

Replace the Cython wrapper (`c-concoct/vbgmm.pyx`) with a PyO3 module that
exposes the Rust VBGMM to the existing Python pipeline. The Python code
(`concoct/`) remains unchanged — it just calls Rust instead of C.

### Phase 5: Python layer migration (current)

Port the Python modules (`input.py`, `transform.py`, `output.py`, `parser.py`)
into the Rust binary. The scripts (`cut_up_fasta.py`, etc.) may remain Python or
be ported depending on need.

#### Correctness standard: end-to-end hash, not bit-exact intermediates

Unlike the C core (Phase 1–3), the Python layer does not have a bit-exact
equivalence requirement against the original. The Python layer uses pandas and
numpy, which use algorithms (e.g. numpy's pairwise summation for column/row sums)
that are impractical to replicate exactly in Rust without importing numpy's C
internals. Minor float divergence (a few ULP) in intermediate values is expected
and acceptable.

The correctness standard for Python→Rust replacements is the **end-to-end
determinism check**: after swapping a Python function for its Rust equivalent,
run `nix build .#checks.x86_64-linux.determinism-small` and `determinism-large`.
If the clustering output hash is unchanged, the swap is correct. If it changes,
investigate — but a hash change does not automatically block the swap if the
cause is understood (e.g. summation order).

**Atomicity matters.** Each replacement must be its own commit. If the hash
changes, the cause is exactly one swap. This makes it possible to bisect and
reason about divergence without ambiguity.

#### Observed: summation order divergence, hash unchanged

Rust's simple left-to-right accumulation for column/row sums produces values
that differ by a few ULP from numpy's 8-way pairwise summation (used by pandas
via numpy). Despite this, both the small (349 contigs) and large (4943 contigs)
determinism checks produced identical clustering hashes after swapping
`load_coverage` from Python to Rust. The intermediate float differences are
absorbed by PCA and VBGMM.

This validates the approach: atomic swaps + end-to-end hash checks catch real
correctness problems while tolerating harmless implementation differences in
glue code.

#### Future: optimisation phase

The same atomic-commit + hash-tracking approach applies to the planned
optimisation phase. After the translation is complete and proven, algorithmic
improvements (e.g. replacing GSL Cholesky, SIMD inner loops, alternative
convergence criteria) will each be their own commit with determinism checks. If
an optimisation changes the hash, the commit message documents the change
explicitly — the reader can see exactly which optimisation altered the output
and decide whether it's acceptable.

## Non-goals

- **Algorithmic improvements**: this rewrite is a *translation*, not a redesign.
  If we want to change the algorithm, that's a separate commit with its own
  justification, after the translation is proven correct.
- **API changes**: the Rust VBGMM must accept and return the same data shapes
  as the C version. API evolution happens after correctness is established.
- **Premature optimisation**: do not add optimisations during a port. If the
  Rust version happens to be faster than C spontaneously (this has been observed
  in practice), that's fine — accept it and move on. If it is *slower*, that is
  a bug and must be investigated before merging. Deliberate optimisation work
  happens after the translation is proven correct, in its own commit with its
  own benchmark proof.

## Phase 0 retrospective

Phase 0 (v0.1.0) established the reproducible baseline. Key findings:

- **The original is deterministic.** Given the same seed and inputs, CONCOCT
  produces identical output regardless of thread count or number of runs. This
  is verified by `determinism-small` and `determinism-large` flake checks with
  known sha256 hashes. The Rust rewrite must preserve this property.

- **Float equivalence is relative to the pinned environment.** Upstream CONCOCT
  does not pin dependency versions. Different sklearn/numpy/scipy versions may
  produce different numerical results in the Python layer (PCA, normalization).
  Our determinism guarantee applies within our nix-pinned environment only. The
  C VBGMM core is the part where bit-exact equivalence matters for the rewrite.

## Known risks and open questions

Identified during Phase 0 review. Resolve before the relevant phase.

**Before Phase 1:**

- Audit C build flags for `-ffast-math` or `-funsafe-math-optimizations` — if
  present, the C oracle is unreliable. (Done: flags are clean.)
- Pin Rust `target-cpu` to `x86-64` baseline, not `native` — AVX can change
  float results across machines.
- Ensure C and Rust link against the same GSL nix derivation.
- Set proptest cases to 10,000+. Filter NaN inputs with `prop_assume!` for C
  functions that don't document NaN handling.
- Define a concrete benchmark threshold for "slower is a failure."

**Before Phase 2:**

- Define what "one change" means for large composite functions (`mstep` is ~180
  lines with BLAS calls). Options: split by logical block, port whole function,
  or refactor the C first.
- Establish policy for bugs found in the original during the rewrite: fix in C
  and re-baseline, fix only in Rust, or adjust the proptest.
- Match thread count between C and Rust for OMP parallel sections. Phase 0
  showed these are embarrassingly parallel, but any parallel float accumulation
  in Rust must preserve the accumulation order.

**Before GSL replacement:**

- GSL Cholesky replacement (nalgebra/faer) will likely diverge on
  ill-conditioned inputs. The replacement commit must document which inputs
  diverge and why that's acceptable, or prove bit-identity.

**Phase 5:**

- Porting the Python glue is a different kind of work than the C core. The
  effort may equal Phases 1-4 combined. Acknowledge this rather than treating
  it as a natural next step.
