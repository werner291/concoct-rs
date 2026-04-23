#!/usr/bin/env python
from nose_compat import assert_equal, assert_true
import numpy as np
import pandas as p
import os
from Bio import SeqIO
from concoct.input import load_composition

# test_normalize_per_contig and test_normalize_per_samples removed:
# normalization is now handled in Rust (src/input.rs load_coverage),
# tested via the determinism checks.

class TestInput(object):
    def test_load_composition(self):
        # Get the directory path of this test file
        f = os.path.dirname(os.path.abspath(__file__))
        # calculate the lengths of the contigs
        seqs = SeqIO.parse("{0}/test_data/composition_some_shortened.fa".format(f),"fasta")
        ids = []
        lengths = []
        for s in seqs:
            if len(s) <= 1000:
                continue
            ids.append(s.id)
            lengths.append(len(s))
        c_len = p.Series(lengths,index=ids,dtype=float)
        # Use load_composition to calculate contig lengths
        composition, contig_lengths = load_composition("{0}/test_data/composition_some_shortened.fa".format(f),4,1000)
        assert_equal(len(c_len), len(contig_lengths))
        # All equal
        for ix in ids:
            assert_equal(c_len.loc[ix], contig_lengths.loc[ix])
