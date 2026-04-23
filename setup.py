#!/usr/bin/env python
from setuptools import setup, find_packages

version = '1.1.0'

setup(name='concoct',
      version=version,
      description="Clustering cONtigs with COverage and ComposiTion",
      long_description="""Concoct is a program that combines three types of
      information - sequence composition, coverage across multiple sample,
      and read-pair linkage - to automatically bin metagenomic contigs
      into genomes. """,
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='Python Scilifelab Metagenomics Binning Clustering Contig',
      author='Brynjar Smari Bjarnason, Johannes Alneberg, Christopher Quince, Anders Andersson, Ino de Bruijn',
      author_email='binni@binnisb.com',
      maintainer='Johannes Alneberg',
      maintainer_email='johannes.alneberg@scilifelab.se',
      url='https://github.com/BinPro/CONCOCT',
      license='FreeBSD',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      scripts=["bin/concoct","bin/concoct_refine", "scripts/cut_up_fasta.py", "scripts/concoct_coverage_table.py",
               "scripts/merge_cutup_clustering.py", "scripts/extract_fasta_bins.py"],
      include_package_data=True,
      zip_safe=False,
      install_requires=['numpy>=1.7.1',
                        'scipy>=0.12.0',
                        'pandas>=0.11.0'],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
