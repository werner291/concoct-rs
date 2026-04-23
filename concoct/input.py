import math
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
    #Coverage import, file has header and contig ids as index
    cov = p.read_table(cov_file, header=0, index_col=0)

    cov = cov[cov.index.isin(contig_lengths.index)]

    # cov_range variable left here for historical reasons. Can be removed entirely
    cov_range = (cov.columns[0],cov.columns[-1])

    # Adding pseudo count
    cov.loc[:,cov_range[0]:cov_range[1]] = cov.loc[:,cov_range[0]:cov_range[1]].add(
            (read_length/contig_lengths),
            axis='index')

    if not no_cov_normalization:
        #Normalize per sample first
        cov.loc[:,cov_range[0]:cov_range[1]] = \
            _normalize_per_sample(cov.loc[:,cov_range[0]:cov_range[1]])

    temp_cov_range = None
    # Total coverage should be calculated after per sample normalization
    if add_total_coverage:
        cov['total_coverage'] = cov.loc[:,cov_range[0]:cov_range[1]].sum(axis=1)
        temp_cov_range = (cov_range[0],'total_coverage')
    
    if not no_cov_normalization:
        # Normalize contigs next
        cov.loc[:,cov_range[0]:cov_range[1]] = \
            _normalize_per_contig(cov.loc[:,cov_range[0]:cov_range[1]])

    if temp_cov_range:
        cov_range = temp_cov_range

    # Log transform
    cov.loc[:,cov_range[0]:cov_range[1]] = \
        cov.loc[:,cov_range[0]:cov_range[1]].map(math.log)

    logging.info('Successfully loaded coverage data.')
    return cov, cov_range
    
def _normalize_per_sample(arr):
    """ Divides respective column of arr with its sum. """
    return arr.divide(arr.sum(axis=0),axis=1)

def _normalize_per_contig(arr):
    """ Divides respective row of arr with its sum. """
    return arr.divide(arr.sum(axis=1),axis=0)
