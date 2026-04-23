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
    // Arg parsing succeeds, binary fails later when opening the file.
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
    // Verify the Rust CLI accepts all of them without an arg-parsing error.
    // The binary will fail when trying to open nonexistent files, but that's
    // expected — we're testing flag parsing, not file I/O.
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
    // clap errors contain "error: " followed by usage info. File-not-found
    // errors are runtime, not arg parsing. Check for clap's signature.
    assert!(
        !stderr.contains("Usage:"),
        "All underscore flags should be accepted by arg parser, got: {stderr}"
    );
}

#[test]
fn short_flags_match_python() {
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
        !stderr.contains("Usage:"),
        "All short flags should be accepted, got: {stderr}"
    );
}

#[test]
fn defaults_match_python() {
    // Parse with --help to verify defaults are shown correctly.
    let output = concoct_bin()
        .args(["--help"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Check default values shown in help match concoct/parser.py
    assert!(stdout.contains("[default: 400]"), "default clusters should be 400");
    assert!(stdout.contains("[default: 4]"), "default kmer_length should be 4");
    assert!(stdout.contains("[default: 1000]"), "default length_threshold should be 1000");
    assert!(stdout.contains("[default: 100]"), "default read_length should be 100");
    assert!(stdout.contains("[default: 90]"), "default total_percentage_pca should be 90");
    assert!(stdout.contains("[default: 1]") , "default seed and threads should be 1");
    assert!(stdout.contains("[default: 500]"), "default iterations should be 500");
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
