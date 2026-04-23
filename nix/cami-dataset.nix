# CAMI I High-Complexity Gold Standard Assembly.
#
# 39,140 contigs, 5 samples. Standard metagenome binning benchmark.
# Sczyrba et al., Nature Methods 2017, doi:10.1038/nmeth.4458
#
# The depth file is in MetaBAT format (mean + variance per sample).
# We convert it to CONCOCT format (mean only, renamed columns).
{ fetchurl, runCommand }:

let
  contigs = fetchurl {
    url = "https://portal.nersc.gov/dna/RD/Metagenome_RD/MetaBAT/Files/CAMI/CAMI_high_GoldStandardAssembly.fasta.gz";
    hash = "sha256-FMSCUBvdmKNPqU98YPClmIqPnMDVI4thhzy733zoaU4=";
  };

  depth = fetchurl {
    url = "https://portal.nersc.gov/dna/RD/Metagenome_RD/MetaBAT/Files/CAMI/depth-high.txt";
    hash = "sha256-t9g39U/Qu4f8TVICE7WcU77Y07JzGP0rpa1JGlRMMj4=";
  };

  # Convert MetaBAT depth to CONCOCT coverage format:
  # Input:  contigName contigLen totalAvgDepth sample1_mean sample1_var sample2_mean ...
  # Output: contig_id  sample_1  sample_2  ...
  coverage = runCommand "cami-coverage" {} ''
    awk -F'\t' 'BEGIN { OFS="\t" }
    NR==1 {
      printf "contig_id";
      n = 0;
      for (i = 4; i <= NF; i += 2) { n++; printf "\tsample_%d", n }
      print "";
      next
    }
    {
      printf "%s", $1;
      for (i = 4; i <= NF; i += 2) { printf "\t%s", $i }
      print ""
    }' ${depth} > $out
  '';
in
{
  inherit contigs coverage;
}
