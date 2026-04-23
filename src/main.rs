mod cli;

use clap::Parser;
use cli::Args;
use concoct::input;
use concoct::output::{self, OutputPaths};
use concoct::pca;
use concoct::vbgmm;

use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::sync::Mutex;
use std::time::SystemTime;

/// Simple file logger matching Python's logging format.
struct Logger {
    file: Mutex<BufWriter<File>>,
}

impl Logger {
    fn new(path: &str) -> Self {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                std::fs::create_dir_all(parent).ok();
            }
        }
        let file = File::create(path).expect("failed to create log file");
        Logger {
            file: Mutex::new(BufWriter::new(file)),
        }
    }

    fn info(&self, msg: &str) {
        self.write("INFO", msg);
    }

    fn error(&self, msg: &str) {
        self.write("ERROR", msg);
    }

    fn warning(&self, msg: &str) {
        self.write("WARNING", msg);
    }

    fn write(&self, level: &str, msg: &str) {
        // Timestamp: seconds since epoch (matches Python's asctime roughly)
        let secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{secs}:{level}:concoct:{msg}");
            let _ = f.flush();
        }
    }
}

fn main() {
    let args = Args::parse();

    if let Err(e) = args.validate() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }

    let seed = args.resolved_seed();
    let paths = OutputPaths::new(&args.basename, args.length_threshold);

    let log = Logger::new(&paths.log_file);

    log.info(&format!(
        "Results created at {}",
        std::fs::canonicalize(&paths.concoct_path)
            .unwrap_or_else(|_| std::path::PathBuf::from(&paths.concoct_path))
            .display()
    ));

    eprintln!(
        "Up and running. Check {} for progress",
        std::fs::canonicalize(&paths.log_file)
            .unwrap_or_else(|_| std::path::PathBuf::from(&paths.log_file))
            .display()
    );

    // Write args
    output::write_args(&paths.args_file, &format!("{args:?}"));

    if args.threads == 1 {
        log.warning(
            "CONCOCT is running in single threaded mode. \
             Please, consider adjusting the --threads parameter.",
        );
    }

    // Load and join composition + coverage
    let comp_file = args
        .composition_file
        .as_deref()
        .expect("composition_file required");
    let comp_reader = BufReader::new(File::open(comp_file).unwrap_or_else(|e| {
        eprintln!("error: {comp_file}: {e}");
        std::process::exit(1);
    }));

    let cov_reader = args.coverage_file.as_deref().map(|path| {
        BufReader::new(File::open(path).unwrap_or_else(|e| {
            eprintln!("error: {path}: {e}");
            std::process::exit(1);
        }))
    });

    let joined = input::load_and_join(
        comp_reader,
        cov_reader,
        args.kmer_length,
        args.length_threshold,
        args.no_cov_normalization,
        !args.no_total_coverage,
        args.read_length as f64,
    );

    if joined.n_contigs < 2 {
        log.error("Not enough contigs pass the threshold filter. Exiting!");
        std::process::exit(-1 & 0xFF); // Match Python's sys.exit(-1) → 255
    }

    // PCA
    let n_components = match args.pca_components() {
        Some(ratio) => ratio,
        None => joined.n_columns as f64,
    };

    let pca_result = pca::pca(
        &joined.data,
        joined.n_contigs,
        joined.n_columns,
        n_components,
    );

    log.info(&format!(
        "Performed PCA, resulted in {} dimensions",
        pca_result.n_components
    ));

    // Write original data
    if !args.no_original_data {
        output::write_original_data(
            &paths.original_file,
            &joined.data,
            joined.n_columns,
            &joined.contig_ids,
            &joined.column_names,
        );
    }

    // Write PCA output
    output::write_pca(
        &paths.pca_file,
        &pca_result.transformed,
        pca_result.n_components,
        &joined.contig_ids,
    );

    output::write_pca_components(
        &paths.pca_components_file,
        &pca_result.components,
        pca_result.n_components,
        pca_result.n_features,
    );

    log.info("PCA transformed data.");

    log.info(&format!(
        "Will call vbgmm with parameters: {}, {}, {}, {}, {}",
        paths.concoct_path,
        args.clusters,
        args.length_threshold,
        args.threads,
        args.iterations
    ));

    // VBGMM clustering
    let assignments = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("failed to build Rayon thread pool")
        .install(|| {
            vbgmm::vbgmm_fit(
                &pca_result.transformed,
                joined.n_contigs,
                pca_result.n_components,
                args.clusters,
                seed,
                args.iterations,
            )
        });

    // Write cluster assignments
    output::write_assign(&paths.assign_file, &assignments, &joined.contig_ids);

    log.info("CONCOCT Finished");
}
