# concoct-rs

A gradual rewrite of [CONCOCT](https://github.com/BinPro/CONCOCT) from Python+C
into Rust.

CONCOCT is a program for unsupervised binning of metagenomic contigs by using
nucleotide composition, coverage data in multiple samples and linkage data from
paired end reads.

## Status

Phase 0 (scaffolding) is complete. The original Python+C codebase builds
reproducibly under nix, all 24 upstream tests pass, and output determinism is
verified. See [METHODOLOGY.md](METHODOLOGY.md) for the rewrite plan.

## Installation

### Docker (recommended for most users)

```bash
nix build .#packages.x86_64-linux.docker
docker load < result
```

Then run concoct via:

```bash
docker run --rm \
  -v /path/to/data:/input:ro \
  -v /path/to/output:/output \
  ghcr.io/werner291/concoct-rs:latest \
  --coverage_file /input/coverage_table.tsv \
  --composition_file /input/contigs.fa \
  --basename /output/
```

### Nix dev shell

Requires [nix](https://nixos.org/) with flakes enabled.

```bash
# Enter the dev shell (via direnv, or manually)
nix develop

# Run all checks
nix flake check

# Run a specific check
nix build .#checks.x86_64-linux.pytest-integration
```

## Basic usage

Cut contigs into smaller parts:

```bash
cut_up_fasta.py original_contigs.fa -c 10000 -o 0 --merge_last -b contigs_10K.bed > contigs_10K.fa
```

Generate coverage depth table. This assumes the directory `mapping/` contains
sorted and indexed BAM files mapped against the original contigs:

```bash
concoct_coverage_table.py contigs_10K.bed mapping/Sample*.sorted.bam > coverage_table.tsv
```

Run concoct:

```bash
concoct --composition_file contigs_10K.fa --coverage_file coverage_table.tsv -b concoct_output/
```

Merge subcontig clustering into original contig clustering:

```bash
merge_cutup_clustering.py concoct_output/clustering_gt1000.csv > concoct_output/clustering_merged.csv
```

Extract bins as individual FASTA:

```bash
mkdir concoct_output/fasta_bins
extract_fasta_bins.py original_contigs.fa concoct_output/clustering_merged.csv --output_path concoct_output/fasta_bins
```

## Checks

| Check | Tests | What it covers |
|-------|-------|----------------|
| `pytest-unit-input` | 5 | K-mer composition, normalization, feature mapping |
| `pytest-cut-up-fasta` | 2 | Contig chunking with overlap and BED output |
| `pytest-gen-input-table` | 2 | Coverage table generation from BAM and BED files |
| `pytest-integration` | 11 | Full concoct pipeline: PCA, VBGMM, output, seeding |
| `pytest-cog-table` | 2 | Single-copy gene annotation via COG table |
| `pytest-merge-cutup` | 1 | Merging chunk-level clusters back to contigs |
| `pytest-integration-scripts` | 1 | End-to-end pipeline with all preprocessing scripts |
| `determinism-small` | - | Output reproducibility on 349-contig dataset |
| `determinism-large` | - | Output reproducibility on 4943-contig dataset |
| `docker` | - | Docker image build, pipeline run, and hash verification in a NixOS VM |

## Methodology

The rewrite follows a strict methodology documented in
[METHODOLOGY.md](METHODOLOGY.md): one commit per change, one reproducible proof
per commit, bit-identical float output, adversarial proptest equivalence tests,
and benchmark comparisons.

## Citation

If you use concoct-rs, **please cite the original CONCOCT paper first**. The
algorithm is theirs; this project is a rewrite of their implementation. If you
are quoting the algorithm itself (the variational Bayesian GMM for contig
binning), the original paper is the authoritative reference.

> Alneberg, J., Bjarnason, B.S., de Bruijn, I., Schirmer, M., Quick, J.,
> Ijaz, U.Z., Lahti, L., Loman, N.J., Andersson, A.F. & Quince, C. Binning
> metagenomic contigs by coverage and composition. *Nat Methods* **11**,
> 1144--1146 (2014). https://doi.org/10.1038/nmeth.3103

You may additionally cite this repository for the Rust rewrite specifically.

This project follows the principles of [rewrites.bio](https://rewrites.bio/):
produce the same results as the original, cite the original, and disclose AI
assistance.

## Related projects

- [maxbin-rs](https://github.com/werner291/maxbin-rs) -- Rust rewrite of MaxBin2
- [rewrites.bio](https://rewrites.bio/) -- a manifesto for bioinformatics rewrites

## Credits

This rewrite is by [Werner Kroneman](https://github.com/werner291), with
assistance from [Claude Code](https://claude.ai/claude-code) (Anthropic).
