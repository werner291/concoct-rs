// CLI argument parsing for concoct.
//
// Matches the interface of concoct/parser.py. All flag names, defaults,
// and validation rules are preserved for drop-in compatibility.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "concoct", version)]
#[clap(rename_all = "snake_case")]
pub struct Args {
    /// Coverage file: table with contigs as rows, samples as columns,
    /// tab-separated average coverage values.
    #[arg(long)]
    pub coverage_file: Option<String>,

    /// Composition file: sequences in FASTA format, used to calculate
    /// k-mer composition (genomic signature) of each contig.
    #[arg(long)]
    pub composition_file: Option<String>,

    /// Maximal number of clusters for VBGMM.
    #[arg(short, long, default_value_t = 400)]
    pub clusters: usize,

    /// K-mer length for composition calculation.
    #[arg(short, long, default_value_t = 4)]
    pub kmer_length: usize,

    /// Number of threads to use.
    #[arg(short, long, default_value_t = 1)]
    pub threads: usize,

    /// Sequence length threshold: contigs shorter than this are excluded.
    #[arg(short, long, default_value_t = 1000)]
    pub length_threshold: usize,

    /// Read length for coverage calculation.
    #[arg(short, long, default_value_t = 100)]
    pub read_length: usize,

    /// Percentage of variance explained by principal components (1-100).
    /// 100 means use all components.
    #[arg(long, default_value_t = 90)]
    pub total_percentage_pca: u32,

    /// Basename for output files or directory. A trailing '/' is
    /// interpreted as a directory. Defaults to current directory.
    #[arg(short, long, default_value = ".")]
    pub basename: String,

    /// Random seed for clustering. 0 gives a random seed, 1 is the
    /// default. Any positive integer can be used.
    #[arg(short, long, default_value_t = 1)]
    pub seed: u64,

    /// Maximum number of VB iterations.
    #[arg(short, long, default_value_t = 500)]
    pub iterations: usize,

    /// Skip coverage normalization (only log-transform).
    #[arg(long, default_value_t = false)]
    pub no_cov_normalization: bool,

    /// Do not add total coverage column.
    #[arg(long, default_value_t = false)]
    pub no_total_coverage: bool,

    /// Do not save original data to disk.
    #[arg(long, default_value_t = false)]
    pub no_original_data: bool,

    /// Write convergence info to files.
    #[arg(short = 'o', long, default_value_t = false)]
    pub converge_out: bool,
}

impl Args {
    /// Resolve the seed value, matching Python's set_random_state:
    /// 0 → random, anything else → use as-is.
    pub fn resolved_seed(&self) -> u64 {
        if self.seed == 0 {
            // Python uses randint(2, 10000); we use a time-based seed
            // for true randomness rather than a small range.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        } else {
            self.seed
        }
    }

    /// PCA components setting, matching Python's logic:
    /// 100% → None (all components), otherwise fraction.
    pub fn pca_components(&self) -> Option<f64> {
        if self.total_percentage_pca >= 100 {
            None
        } else {
            Some(self.total_percentage_pca as f64 / 100.0)
        }
    }

    /// Validate that at least one input file is provided.
    pub fn validate(&self) -> Result<(), String> {
        if self.coverage_file.is_none() && self.composition_file.is_none() {
            return Err(
                "No input data supplied, add file(s) using \
                 --coverage_file <cov_file> and/or \
                 --composition_file <comp_file>"
                    .to_string(),
            );
        }
        Ok(())
    }
}
