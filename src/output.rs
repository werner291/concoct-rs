// Output file writing for CONCOCT results.
//
// Ports concoct/output.py. Matches the exact CSV format produced by
// the original Python code, including the %1.8e float format.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

extern "C" {
    fn snprintf(buf: *mut u8, size: usize, fmt: *const i8, ...) -> i32;
}

/// Format a float as %1.8e, matching C printf / Python '%1.8e' exactly.
fn format_scientific(v: f64) -> String {
    let mut buf = [0u8; 64];
    let n = unsafe { snprintf(buf.as_mut_ptr(), 64, b"%1.8e\0".as_ptr() as *const i8, v) };
    String::from_utf8_lossy(&buf[..n as usize]).to_string()
}

/// Output path configuration, matching concoct/output.py Output.__init__.
pub struct OutputPaths {
    pub concoct_path: String,
    pub args_file: String,
    pub original_file: String,
    pub pca_file: String,
    pub assign_file: String,
    pub pca_components_file: String,
    pub log_file: String,
}

impl OutputPaths {
    /// Create output paths from basename and threshold.
    /// Matches the Python Output.__init__ path logic exactly.
    pub fn new(basename: &str, threshold: usize) -> Self {
        let concoct_path = if Path::new(basename).is_dir() {
            if basename.ends_with('/') {
                basename.to_string()
            } else {
                format!("{basename}/")
            }
        } else if basename.ends_with('/') {
            let abs = fs::canonicalize(basename.trim_end_matches('/'))
                .unwrap_or_else(|_| {
                    fs::create_dir_all(basename).unwrap();
                    fs::canonicalize(basename).unwrap()
                });
            format!("{}/", abs.display())
        } else {
            format!("{basename}_")
        };

        let t = threshold;
        OutputPaths {
            args_file: format!("{concoct_path}args.txt"),
            original_file: format!("{concoct_path}original_data_gt{t}.csv"),
            pca_file: format!("{concoct_path}PCA_transformed_data_gt{t}.csv"),
            assign_file: format!("{concoct_path}clustering_gt{t}.csv"),
            pca_components_file: format!("{concoct_path}PCA_components_data_gt{t}.csv"),
            log_file: format!("{concoct_path}log.txt"),
            concoct_path,
        }
    }
}

/// Write the args.txt file.
pub fn write_args(path: &str, args_str: &str) {
    let mut f = File::create(path).expect("failed to create args.txt");
    writeln!(f, "{args_str}").unwrap();
}

/// Write cluster assignments as CSV.
pub fn write_assign(path: &str, assignments: &[i32], contig_ids: &[String]) {
    let mut w = BufWriter::new(File::create(path).expect("failed to create assign file"));
    writeln!(w, "contig_id,cluster_id").unwrap();
    for (cid, &cluster) in contig_ids.iter().zip(assignments) {
        writeln!(w, "{cid},{cluster}").unwrap();
    }
}

/// Write PCA-transformed data as CSV.
pub fn write_pca(
    path: &str,
    data: &[f64],
    n_components: usize,
    contig_ids: &[String],
) {
    let mut w = BufWriter::new(File::create(path).expect("failed to create PCA file"));
    // Header: contig_id,0,1,...,n_components-1
    write!(w, "contig_id").unwrap();
    for i in 0..n_components {
        write!(w, ",{i}").unwrap();
    }
    writeln!(w).unwrap();
    // Data rows
    for (i, cid) in contig_ids.iter().enumerate() {
        write!(w, "{cid}").unwrap();
        for j in 0..n_components {
            write!(w, ",{}", format_scientific(data[i * n_components + j])).unwrap();
        }
        writeln!(w).unwrap();
    }
}

/// Write PCA component matrix as CSV (no row index).
pub fn write_pca_components(
    path: &str,
    components: &[f64],
    n_components: usize,
    n_features: usize,
) {
    let mut w = BufWriter::new(File::create(path).expect("failed to create PCA components file"));
    for i in 0..n_components {
        for j in 0..n_features {
            if j > 0 {
                write!(w, ",").unwrap();
            }
            write!(w, "{}", format_scientific(components[i * n_features + j])).unwrap();
        }
        writeln!(w).unwrap();
    }
}

/// Write original joined data as CSV.
pub fn write_original_data(
    path: &str,
    data: &[f64],
    n_columns: usize,
    contig_ids: &[String],
    column_names: &[String],
) {
    let mut w = BufWriter::new(File::create(path).expect("failed to create original data file"));
    // Header: empty first column, then column names
    write!(w, "").unwrap();
    for name in column_names {
        write!(w, ",{name}").unwrap();
    }
    writeln!(w).unwrap();
    // Data rows
    for (i, cid) in contig_ids.iter().enumerate() {
        write!(w, "{cid}").unwrap();
        for j in 0..n_columns {
            write!(w, ",{}", format_scientific(data[i * n_columns + j])).unwrap();
        }
        writeln!(w).unwrap();
    }
}
