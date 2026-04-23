#!/usr/bin/env python
from nose_compat import assert_equal
import os
from Bio import SeqIO
import vbgmm


class TestInput(object):
    def test_load_composition(self):
        """Verify Rust load_composition returns correct contig lengths."""
        f = os.path.dirname(os.path.abspath(__file__))
        comp_file = "{0}/test_data/composition_some_shortened.fa".format(f)

        # Reference: contig lengths from BioPython
        expected = {}
        for s in SeqIO.parse(comp_file, "fasta"):
            if len(s) > 1000:
                expected[s.id] = float(len(s))

        # Rust load_composition via PyO3
        data, contig_ids, contig_lengths = vbgmm.load_composition(comp_file, 4, 1000)
        assert_equal(len(expected), len(contig_ids))

        for cid, length in zip(contig_ids, contig_lengths):
            assert_equal(expected[cid], length)
