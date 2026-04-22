# concoct-rs

A reproducible repackaging of [CONCOCT](https://github.com/BinPro/CONCOCT),
the metagenomic contig binner.

CONCOCT bins metagenomic contigs using nucleotide composition, coverage data
across multiple samples, and read-pair linkage. This project wraps the original
Python+C implementation in a nix flake for reproducible builds, fixes
compatibility issues with modern dependencies, and provides a Docker image.

## Installation

### Docker

```bash
docker pull ghcr.io/werner291/concoct-rs:v0.1.0
```

Or build from source:

```bash
nix build .#packages.x86_64-linux.docker
docker load < result
```

Then run:

```bash
docker run --rm \
  -v /path/to/data:/input:ro \
  -v /path/to/output:/output \
  ghcr.io/werner291/concoct-rs:latest \
  --coverage_file /input/coverage_table.tsv \
  --composition_file /input/contigs.fa \
  --basename /output/
```

### Nix

```bash
nix build              # build the concoct package
./result/bin/concoct --help

nix develop            # dev shell with all dependencies
nix flake check        # run all 24 tests + determinism checks + Docker VM test
```

## What this project fixes

The upstream CONCOCT 1.1.0 has
[30+ open issues](https://github.com/BinPro/CONCOCT/issues), many of which
are installation failures and dependency breakage. This repackaging addresses:

- **sklearn 1.8 compatibility** (issues #338, #323, #322, #321): adapted PCA
  code for changes in DataFrame column type handling.
- **nose test framework** (end-of-life): migrated to pytest.
- **Dependency pinning**: all dependencies pinned via nix flake, including GSL,
  Python scientific stack, bedtools, samtools, and bcbio-gff.
- **Missing test data**: integration test data fetched as a pinned nix
  derivation from BinPro/integration_test_data.
- **Output determinism**: verified across thread counts and repeated runs on
  two datasets.
- **CI**: GitHub Actions replaces the defunct Travis CI configuration.

The VBGMM clustering algorithm is untouched.

## Basic usage

Cut contigs into smaller parts:

```bash
cut_up_fasta.py original_contigs.fa -c 10000 -o 0 --merge_last -b contigs_10K.bed > contigs_10K.fa
```

Generate coverage depth table (assumes `mapping/` contains sorted and indexed
BAM files):

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

## Rust rewrite (experimental)

The `work` branch contains a bit-identical Rust reimplementation of the VBGMM
core (c_vbgmm_fit.c). It matches C performance at production sizes and has
proptest equivalence tests for every function. See
[METHODOLOGY.md](METHODOLOGY.md) for the approach. The Rust port is not
shipped — Phase 0 (this branch) is the released product.

## Citation

If you use concoct-rs, **please cite the original CONCOCT paper first**. The
algorithm is theirs; this project is a repackaging of their implementation.

> Alneberg, J., Bjarnason, B.S., de Bruijn, I., Schirmer, M., Quick, J.,
> Ijaz, U.Z., Lahti, L., Loman, N.J., Andersson, A.F. & Quince, C. Binning
> metagenomic contigs by coverage and composition. *Nat Methods* **11**,
> 1144--1146 (2014). https://doi.org/10.1038/nmeth.3103

This project follows the principles of [rewrites.bio](https://rewrites.bio/):
produce the same results as the original, cite the original, and disclose AI
assistance.

## Related projects

- [maxbin-rs](https://github.com/werner291/maxbin-rs) -- Rust rewrite of MaxBin2
- [rewrites.bio](https://rewrites.bio/) -- a manifesto for bioinformatics rewrites

## Credits

This repackaging is by [Werner Kroneman](https://github.com/werner291), with
assistance from [Claude Code](https://claude.ai/claude-code) (Anthropic).
