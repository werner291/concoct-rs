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

### Phase 1: Leaf functions (current)

Set up the Rust crate skeleton (cargo workspace, criterion, proptest), build the
C library as a standalone `.a`/`.so` for FFI from Rust tests, then port the leaf
functions:

Port the leaf functions from `c_vbgmm_fit.c` that have no dependencies on other
CONCOCT functions:

- `calcDist` (L1300-1311): Euclidean distance. Trivial, good first target.
- `calcSampleVar` (L202-227): Sample mean and variance per dimension.
- `decomposeMatrix` (L507-528): Cholesky decomposition + inversion via GSL.
- `dLogWishartB` (L1207-1236): Log Wishart normalisation constant.
- `dWishartExpectLogDet` (L1238-1257): Expected log determinant of Wishart.
- `updateMeans` (L1259-1298): Weighted cluster means from assignments.

Each function gets: one proptest, one benchmark, one commit.

### Phase 2: Composite functions

Port the functions that compose leaf functions:

- `mstep` (L530-709): M-step for a single component (calls GSL BLAS).
- `calcZ_MP` (L979-1046): E-step / responsibility calculation.
- `calcVBL_MP` (L895-977): Variational lower bound (calls `eqnA`, `eqnB`).
- `performMStepMP` (L711-751): Parallel M-step across components.
- `initKMeans` (L753-821): K-means initialisation.

### Phase 3: Training loop and driver

- `gmmTrainVB_MP` (L1048-1114): The EM/VB iteration loop.
- `fitEM_MP` (L397-427): Top-level fit orchestration.
- `compressCluster` (L429-505): Post-fit cluster compression.
- `driverMP` (L51-155): The full driver.

At this point, the Rust VBGMM can replace the C extension entirely.

### Phase 4: Python integration layer

Replace the Cython wrapper (`c-concoct/vbgmm.pyx`) with a PyO3 module that
exposes the Rust VBGMM to the existing Python pipeline. The Python code
(`concoct/`) remains unchanged — it just calls Rust instead of C.

### Phase 5: Python layer migration

Port the Python modules (`input.py`, `transform.py`, `output.py`, `parser.py`)
into the Rust binary. The scripts (`cut_up_fasta.py`, etc.) may remain Python or
be ported depending on need.

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
