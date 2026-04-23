# -*- coding: utf-8 -*-
"""
Output writing for CONCOCT results.

Replaces the pandas-based Output class with direct CSV writing.
Float format matches the original: %1.8e (C printf / Python % format).
"""
from __future__ import print_function
import os
import sys
import logging


class Output(object):
    CONCOCT_PATH = None
    ARGS_FILE = None
    PCA_FILE_BASE = None
    PCA_COMPONENTS_FILE_BASE = None
    FLOAT_FORMAT = '%1.8e'

    @classmethod
    def __init__(self, basename, args):
        if os.path.isdir(basename):
            if basename[-1] == '/':
                self.CONCOCT_PATH = basename
            else:
                self.CONCOCT_PATH = basename + '/'
        elif basename[-1] == '/':
            basename_path = os.path.abspath(basename)
            os.mkdir(basename_path)
            self.CONCOCT_PATH = basename_path + '/'
        else:
            basename_path = os.path.abspath(basename)
            self.CONCOCT_PATH = basename + '_'

        self.ARGS_FILE = self.CONCOCT_PATH + "args.txt"
        self.ORIGINAL_FILE_BASE = self.CONCOCT_PATH + "original_data_gt{0}.csv"
        self.PCA_FILE_BASE = self.CONCOCT_PATH + \
            "PCA_transformed_data_gt{0}.csv"
        self.ASSIGN_FILE_BASE = self.CONCOCT_PATH + \
            "clustering_gt{0}.csv"
        self.PCA_COMPONENTS_FILE_BASE = self.CONCOCT_PATH + \
            "PCA_components_data_gt{0}.csv"
        self.LOG_FILE_BASE = self.CONCOCT_PATH + 'log.txt'

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=self.LOG_FILE_BASE,
            level=logging.INFO,
            filemode='w',
            format='%(asctime)s:%(levelname)s:%(name)s:%(message)s'
            )

        logging.info("Results created at {0}".format(
            os.path.abspath(self.CONCOCT_PATH)))

        print("Up and running. Check {0} for progress".format(
                        os.path.abspath(self.LOG_FILE_BASE)
                    ), file=sys.stderr)

        with open(self.ARGS_FILE, "w+") as fh:
            print(args, file=fh)

    @classmethod
    def write_pca(self, transform, threshold, contig_ids):
        """Write PCA-transformed data as CSV.

        transform: 2-D array (n_contigs × n_components)
        contig_ids: list of contig ID strings
        """
        n_components = transform.shape[1] if hasattr(transform, 'shape') else len(transform[0])
        path = self.PCA_FILE_BASE.format(threshold)
        with open(path, 'w') as f:
            # Header: contig_id,0,1,...,n_components-1
            f.write('contig_id,' + ','.join(str(i) for i in range(n_components)) + '\n')
            for i, cid in enumerate(contig_ids):
                row = transform[i]
                vals = ','.join(self.FLOAT_FORMAT % v for v in row)
                f.write(f'{cid},{vals}\n')
        logging.info('Wrote PCA transformed file.')

    @classmethod
    def write_assign(self, assign, threshold, contig_ids):
        """Write cluster assignments as CSV.

        assign: 1-D array of int32 cluster IDs
        contig_ids: list of contig ID strings
        """
        path = self.ASSIGN_FILE_BASE.format(threshold)
        with open(path, 'w') as f:
            f.write('contig_id,cluster_id\n')
            for cid, cluster in zip(contig_ids, assign):
                f.write(f'{cid},{int(cluster)}\n')
        logging.info('Wrote assign file.')

    @classmethod
    def write_pca_components(self, components, threshold):
        """Write PCA component matrix as CSV (no index).

        components: 2-D array (n_components × n_features)
        """
        path = self.PCA_COMPONENTS_FILE_BASE.format(threshold)
        with open(path, 'w') as f:
            for row in components:
                f.write(','.join(self.FLOAT_FORMAT % v for v in row) + '\n')
        logging.info('Wrote PCA components file.')

    @classmethod
    def write_original_data(self, data, threshold, contig_ids, column_names):
        """Write original joined data as CSV.

        data: 2-D array (n_contigs × n_columns)
        contig_ids: list of contig ID strings
        column_names: list of column name strings
        """
        path = self.ORIGINAL_FILE_BASE.format(threshold)
        with open(path, 'w') as f:
            # Header: empty first column (index), then column names
            f.write(',' + ','.join(column_names) + '\n')
            for i, cid in enumerate(contig_ids):
                row = data[i]
                vals = ','.join(self.FLOAT_FORMAT % v for v in row)
                f.write(f'{cid},{vals}\n')
        logging.info('Wrote original filtered data file.')
