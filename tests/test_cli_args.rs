// CLI argument parsing tests.
//
// Verifies that the Rust CLI accepts the same flags and defaults as
// the Python argparse in concoct/parser.py.

use std::process::Command;

fn concoct_bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_concoct"))
}

#[test]
fn no_input_files_exits_with_error() {
    let output = concoct_bin().output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("No input data supplied"),
        "Expected input validation error, got: {stderr}"
    );
}

#[test]
fn coverage_file_only_is_accepted() {
    // Should parse successfully (will fail later when it tries to open
    // the file, but arg parsing itself should succeed).
    let output = concoct_bin()
        .args(["--coverage_file", "nonexistent.tsv"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("No input data supplied"),
        "Should accept coverage_file alone, got: {stderr}"
    );
}

#[test]
fn composition_file_only_is_accepted() {
    let output = concoct_bin()
        .args(["--composition_file", "nonexistent.fa"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("No input data supplied"),
        "Should accept composition_file alone, got: {stderr}"
    );
}

#[test]
fn underscore_flags_are_accepted() {
    // Python uses underscore-separated flags (--coverage_file, not --coverage-file).
    // Verify the Rust CLI matches.
    let output = concoct_bin()
        .args([
            "--coverage_file", "cov.tsv",
            "--composition_file", "comp.fa",
            "--no_cov_normalization",
            "--no_total_coverage",
            "--no_original_data",
            "--total_percentage_pca", "80",
            "--kmer_length", "5",
            "--length_threshold", "500",
            "--read_length", "150",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("error"),
        "All underscore flags should be accepted, got: {stderr}"
    );
}

#[test]
fn short_flags_match_python() {
    // Python short flags: -c, -k, -t, -l, -r, -b, -s, -i, -o, -d, -v
    let output = concoct_bin()
        .args([
            "--coverage_file", "cov.tsv",
            "-c", "10",
            "-k", "5",
            "-t", "4",
            "-l", "500",
            "-r", "150",
            "-b", "output/",
            "-s", "42",
            "-i", "200",
            "-o",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("error"),
        "All short flags should be accepted, got: {stderr}"
    );
}

#[test]
fn defaults_match_python() {
    // Parse with minimal args, check defaults via debug output.
    let output = concoct_bin()
        .args(["--coverage_file", "cov.tsv"])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Check defaults match concoct/parser.py
    assert!(stderr.contains("clusters: 400"), "default clusters should be 400");
    assert!(stderr.contains("kmer_length: 4"), "default kmer_length should be 4");
    assert!(stderr.contains("threads: 1"), "default threads should be 1");
    assert!(stderr.contains("length_threshold: 1000"), "default length_threshold should be 1000");
    assert!(stderr.contains("read_length: 100"), "default read_length should be 100");
    assert!(stderr.contains("total_percentage_pca: 90"), "default total_percentage_pca should be 90");
    assert!(stderr.contains("seed: 1"), "default seed should be 1");
    assert!(stderr.contains("iterations: 500"), "default iterations should be 500");
    assert!(stderr.contains("no_cov_normalization: false"), "default no_cov_normalization should be false");
    assert!(stderr.contains("no_total_coverage: false"), "default no_total_coverage should be false");
    assert!(stderr.contains("no_original_data: false"), "default no_original_data should be false");
    assert!(stderr.contains("converge_out: false"), "default converge_out should be false");
}

#[test]
fn version_flag() {
    let output = concoct_bin().arg("--version").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("concoct"),
        "Version output should contain program name, got: {stdout}"
    );
}
