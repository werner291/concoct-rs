import logging

import pandas as p
import vbgmm


def load_data(args):
    composition, contig_lengths = load_composition(
        args.composition_file,
        args.kmer_length,
        args.length_threshold
        )

    if args.coverage_file:
        cov, cov_range = load_coverage(
            args.coverage_file,
            contig_lengths,
            args.no_cov_normalization,
            add_total_coverage = (not args.no_total_coverage),
            read_length = args.read_length
            )
    else:
        cov, cov_range = None, None

    return composition, cov, cov_range


def load_composition(comp_file, kmer_len, threshold):
    # Rust handles FASTA parsing, k-mer counting, normalization, and
    # log-transform (src/input.rs load_composition).
    data, contig_ids, contig_lengths = vbgmm.load_composition(
            comp_file, kmer_len, threshold)
    composition = p.DataFrame(data, index=contig_ids, dtype=float)
    contig_lengths = p.Series(
            dict(zip(contig_ids, contig_lengths)), dtype=float)

    logging.info('Successfully loaded composition data.')
    return composition, contig_lengths

def load_coverage(cov_file, contig_lengths, no_cov_normalization, add_total_coverage=False, read_length=100):
    # Rust handles TSV parsing, filtering, pseudo-count, normalization,
    # and log-transform (src/input.rs load_coverage).
    data, filtered_ids, col_names, range_start, range_end = \
        vbgmm.load_coverage_rs(
            cov_file,
            list(contig_lengths.index),
            list(contig_lengths.values),
            no_cov_normalization,
            add_total_coverage,
            float(read_length))

    cov = p.DataFrame(data, index=filtered_ids, columns=col_names, dtype=float)
    cov_range = (range_start, range_end)

    logging.info('Successfully loaded coverage data.')
    return cov, cov_range
