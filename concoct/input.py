# Input loading functions.
#
# The driver (bin/concoct) now calls vbgmm.load_and_join() directly.
# These functions remain for backwards compatibility with external scripts.

import logging
import vbgmm


def load_composition(comp_file, kmer_len, threshold):
    """Load composition from a FASTA file.

    Returns (data, contig_ids, contig_lengths) as numpy arrays/lists.
    """
    data, contig_ids, contig_lengths = vbgmm.load_composition(
            comp_file, kmer_len, threshold)
    logging.info('Successfully loaded composition data.')
    return data, contig_ids, contig_lengths
