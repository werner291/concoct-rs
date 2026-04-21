# concoct-rs — CLAUDE.md

## Mandatory: read METHODOLOGY.md first

Before doing any work in this repository, read `METHODOLOGY.md` in full. Every
instruction below enforces that methodology. If there is a conflict between this
file and METHODOLOGY.md, ask the user — do not silently pick one.

## Commit discipline

### Every commit must contain exactly one change and one proof

Do NOT batch unrelated changes into a single commit. Do NOT commit a change
without a corresponding test. If a commit introduces a ported function, it must
also introduce the proptest and/or benchmark that proves it.

Before creating any commit, verify:

1. **Single change**: the diff addresses exactly one thing (one function ported,
   one interface introduced, one dependency replaced). If the diff does two
   things, split it into two commits.
2. **Proof exists**: there is a test that exercises the change, runnable via a
   single `nix` command. Cite this command in the commit message.
3. **Tests pass**: run the specific `nix build .#checks.x86_64-linux.<name>`
   for this commit's proof before committing. Do not commit with a failing proof.

### Commit message format

```
<imperative summary of what changed>

<paragraph explaining WHY this change was made, what effect it has on the
overall system, and how it fits into the rewrite progression>

<citation of the C source location being ported, if applicable,
e.g. c-concoct/c_vbgmm_fit.c:202-227>

<citation of the algorithm reference, if applicable,
e.g. Bishop PRML Equation 10.60>

Proof: nix build .#checks.x86_64-linux.<test-name>
```

Do not write vague commit messages like "port function" or "add tests". The
message must be specific enough that a reader can understand the change, its
motivation, and how to verify it without reading the diff.

### History rewriting

Interactive rebase and history rewriting (amend, squash, reorder) are explicitly
allowed and encouraged when they improve traceability. Examples:

- Squashing a "fix typo" commit into the commit it fixes.
- Reordering commits so the logical progression is clearer.
- Rewording a commit message to add a missing proof citation.

Always confirm with the user before force-pushing to a shared branch.

## Float correctness

### Bit-identical output is mandatory

The Rust implementation must produce **bit-for-bit identical** output to the C
original for the same inputs. Use `f64::to_bits()` comparison in tests, never
approximate equality.

### Known float hazards — check for all of these during every port

1. **Operation order**: any sequence of float operations (sums, products,
   divisions, compound expressions) must execute in the same order as the C
   code. This applies to all arithmetic, not just sums — maxbin-rs was bitten
   by multiplication order specifically. Do not use iterators that may reorder
   (e.g., `par_iter`). Do not use `fold` where the C code uses an explicit loop
   with index-ordered operations.
2. **FMA and operation grouping**: `a * b + c` is not the same as `(a * b) + c`
   if the compiler emits a fused multiply-add. Use explicit parentheses. Use
   `#[inline(never)]` on critical functions during the equivalence-proving
   phase to prevent the compiler from reordering across call boundaries.
3. **Printf format matching**: when the Rust code produces text output, match
   the exact C `printf` format specifier (field width, precision, padding).

When porting a function, add a comment at the top of the Rust function citing
the C source file and line range, and noting any float-sensitive operation
patterns.

### Removing inline bans

`#[inline(never)]` annotations exist to prevent the compiler from reordering
float operations across call boundaries during equivalence verification. They
may be removed **only after** bit-exact equivalence has been established and
committed. Removing them is its own commit, with its own re-run of the
equivalence test as proof that inlining does not change the output. Watch for
this during history rewriting — do not squash an inline-ban removal into the
commit that establishes equivalence, as that defeats the purpose.

## Rewrite order

Follow the inside-out order defined in METHODOLOGY.md:

1. Leaf functions (calcDist, calcSampleVar, decomposeMatrix, etc.)
2. Composite functions (mstep, calcZ_MP, calcVBL_MP, etc.)
3. Training loop and driver
4. Python integration (PyO3 replacing Cython)
5. Python layer migration

Do NOT skip ahead. Do not port a composite function before its leaf dependencies
are ported, tested, and committed. If you believe the order should change, say so
and explain why — do not silently deviate.

## Test requirements

### Proptest equivalence tests

- Generate **adversarial** inputs, not just uniform random. Every proptest input
  strategy must include: special floats (±0.0, subnormals, ±INF, NaN, MAX, MIN,
  EPSILON), near-equal values (1 ULP apart), mixed magnitudes, exact duplicates,
  and boundary sizes (empty, single-element, power-of-two).
- Call both the C function (via FFI) and the Rust function.
- Compare outputs with `f64::to_bits()` (not approximate equality).
- Each ported function gets its own named test target.

### Benchmark comparisons

- Use criterion for benchmarks.
- Run both C and Rust implementations on the same fixed dataset.
- Slower Rust performance is considered a failure — investigate before merging.
- Each ported function gets its own named benchmark target.

### Reproducibility

All tests must be runnable via `nix flake check` or a specific `nix build`
command. No ambient state. No "works on my machine". If a test requires data,
that data is either checked in or fetched as a fixed-output derivation.

## Self-checks

Periodically (especially before commits and at the start of sessions), review
recent work against this checklist:

- [ ] Does every commit in the current branch have exactly one change?
- [ ] Does every commit cite its proof command?
- [ ] Are there any uncommitted changes that should be split?
- [ ] Is the rewrite order being followed (no skipped layers)?
- [ ] Do all proptest comparisons use bit-exact checks?
- [ ] Does each commit's cited `nix build .#checks...` command pass?
- [ ] Would the commit history make sense to a reader who hasn't seen the
      conversation?

If any answer is no, fix it before proceeding. History rewriting is allowed to
fix these issues retroactively.

## Nix

This project uses a nix flake for all builds and tests. The dev shell is
activated via direnv (`use flake` in `.envrc`). All test and benchmark targets
must be defined as flake checks so they are reproducible.

Run individual tests with `nix build .#checks.x86_64-linux.<name>`. Each commit
must cite the specific check that proves it — `nix flake check` is too global to
serve as a proof.

## Proactive critique

Proactive critiques of approach, methodology violations, commit hygiene, and
design tensions are explicitly encouraged. Do not wait to be asked — if
something violates the methodology or looks like it should be split, raise it
immediately.

## What NOT to do

- Do not commit without a proof. Ever.
- Do not use approximate float comparison (epsilon, `assert_relative_eq`, etc.)
  in equivalence tests. Bit-exact only.
- Do not port multiple functions in one commit.
- Do not skip layers in the rewrite order without explicit discussion.
- Do not add algorithmic improvements during a port. Translation first,
  optimisation later, each in its own commit.
- Do not let tests depend on ambient system state. Nix or nothing.
