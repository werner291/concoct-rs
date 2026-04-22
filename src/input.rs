// Input loading: composition (FASTA + k-mer) and coverage (TSV).
//
// Ports concoct/input.py. Each function cites the Python source it replaces.

use std::collections::HashMap;
use std::io::BufRead;

/// A FASTA record: id (first word after '>') and sequence bytes.
pub struct FastaRecord {
    pub id: String,
    pub seq: Vec<u8>,
}

/// Parse FASTA records from a reader.
/// Matches BioPython's SeqIO.parse(file, "fasta") behavior:
/// - id is the first whitespace-delimited token after '>'
/// - sequence lines are stripped of leading/trailing whitespace
/// - blank lines are ignored
/// - case is preserved (caller uppercases if needed)
pub fn parse_fasta<R: BufRead>(reader: R) -> Vec<FastaRecord> {
    let mut records = Vec::new();
    let mut current_id: Option<String> = None;
    let mut current_seq: Vec<u8> = Vec::new();

    for line in reader.lines() {
        let line = line.expect("error reading FASTA");
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(header) = trimmed.strip_prefix('>') {
            if let Some(id) = current_id.take() {
                records.push(FastaRecord { id, seq: current_seq });
                current_seq = Vec::new();
            }
            current_id = Some(
                header.split_whitespace().next().unwrap_or("").to_string()
            );
        } else {
            current_seq.extend_from_slice(trimmed.as_bytes());
        }
    }
    if let Some(id) = current_id {
        records.push(FastaRecord { id, seq: current_seq });
    }

    records
}

/// Result of composition calculation for a single contig.
pub struct ContigComposition {
    pub id: String,
    pub length: usize,
    /// K-mer frequency vector with pseudo count (+1), length = nr_features.
    pub counts: Vec<f64>,
}

/// Calculate k-mer composition from FASTA records.
/// Contigs with length <= threshold are skipped.
///
/// Ports: concoct/input.py _calculate_composition (L33-63)
///
/// Returns composition rows (one per contig that passes the threshold).
pub fn calculate_composition<R: BufRead>(
    reader: R,
    length_threshold: usize,
    kmer_len: usize,
) -> Vec<ContigComposition> {
    let (feature_mapping, nr_features) = generate_feature_mapping(kmer_len);
    let records = parse_fasta(reader);
    let mut result = Vec::new();

    for record in records {
        let seq_len = record.seq.len();
        if seq_len <= length_threshold {
            continue;
        }

        // Count k-mers, matching Python's window() + bincount pattern.
        // K-mers containing non-ATGC bases are skipped (the `if kmer_tuple
        // in feature_mapping` filter in the Python).
        let mut counts = vec![0u64; nr_features];
        if seq_len >= kmer_len {
            let seq_upper: Vec<u8> = record.seq.iter().map(|b| b.to_ascii_uppercase()).collect();
            for window in seq_upper.windows(kmer_len) {
                if let Some(&idx) = feature_mapping.get(window) {
                    counts[idx] += 1;
                }
            }
        }

        // Add pseudo count of 1 (matching `composition_v + np.ones(nr_features)`)
        let counts_f64: Vec<f64> = counts.iter().map(|&c| c as f64 + 1.0).collect();

        result.push(ContigComposition {
            id: record.id,
            length: seq_len,
            counts: counts_f64,
        });
    }

    result
}

/// Load composition from FASTA: calculate k-mer frequencies, normalize
/// per-contig, and log-transform.
///
/// Ports: concoct/input.py load_composition (L65-78)
///
/// The normalization is: log(count_ij / row_sum_i) for each contig i
/// and feature j. The counts already include the +1 pseudo count from
/// calculate_composition.
///
/// Returns (composition matrix as flat row-major Vec<f64> with shape
/// n_contigs × nr_features, contig ids, contig lengths, nr_features).
pub struct CompositionData {
    /// Flat row-major matrix, n_contigs × n_features.
    pub data: Vec<f64>,
    /// Contig IDs, in order.
    pub contig_ids: Vec<String>,
    /// Contig lengths, in order.
    pub contig_lengths: Vec<usize>,
    /// Number of features (columns).
    pub n_features: usize,
}

pub fn load_composition<R: BufRead>(
    reader: R,
    kmer_len: usize,
    length_threshold: usize,
) -> CompositionData {
    let contigs = calculate_composition(reader, length_threshold, kmer_len);
    let n_features = if contigs.is_empty() {
        let (_, nf) = generate_feature_mapping(kmer_len);
        nf
    } else {
        contigs[0].counts.len()
    };

    let mut data = Vec::with_capacity(contigs.len() * n_features);
    let mut contig_ids = Vec::with_capacity(contigs.len());
    let mut contig_lengths = Vec::with_capacity(contigs.len());

    for contig in &contigs {
        // Row sum (matches numpy sum(axis=1) — left-to-right accumulation)
        let row_sum: f64 = contig.counts.iter().sum();

        // log(count / row_sum) for each feature.
        //
        // NOTE: Rust's f64::ln() can differ from numpy's np.log() by 1 ULP.
        // Numpy links Intel SVML (__svml_log8_ha) which is a different
        // implementation from both Rust's ln() and libc's log().
        // Confirmed 2026-04-23: libm log and math.log agree with Rust,
        // numpy disagrees by 1 ULP on specific inputs (e.g. 120/20627).
        // End-to-end output hash is the real equivalence proof.
        for &count in &contig.counts {
            data.push((count / row_sum).ln());
        }

        contig_ids.push(contig.id.clone());
        contig_lengths.push(contig.length);
    }

    CompositionData {
        data,
        contig_ids,
        contig_lengths,
        n_features,
    }
}

/// Generate a mapping from k-mer tuples to canonical feature indices.
/// Reverse complement k-mers map to the same index.
///
/// Ports: concoct/input.py generate_feature_mapping (L129-139)
///
/// Returns (mapping, nr_features) where mapping maps k-mer byte strings
/// to their canonical index.
pub fn generate_feature_mapping(kmer_len: usize) -> (HashMap<Vec<u8>, usize>, usize) {
    let complement = |b: u8| -> u8 {
        match b {
            b'A' => b'T',
            b'T' => b'A',
            b'G' => b'C',
            b'C' => b'G',
            _ => unreachable!(),
        }
    };

    let mut kmer_hash: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut counter = 0usize;

    // Iterate over all k-mers in the same order as Python's
    // itertools.product("ATGC", repeat=kmer_len).
    // Rightmost position cycles fastest: AA, AT, AG, AC, TA, ...
    let bases = [b'A', b'T', b'G', b'C'];
    let n_kmers = 4usize.pow(kmer_len as u32);

    for i in 0..n_kmers {
        let kmer: Vec<u8> = (0..kmer_len)
            .map(|pos| {
                let digit = (i / 4usize.pow((kmer_len - 1 - pos) as u32)) % 4;
                bases[digit]
            })
            .collect();

        if !kmer_hash.contains_key(&kmer) {
            kmer_hash.insert(kmer.clone(), counter);
            let rev_compl: Vec<u8> = kmer.iter().rev().map(|&b| complement(b)).collect();
            kmer_hash.insert(rev_compl, counter);
            counter += 1;
        }
    }

    (kmer_hash, counter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fasta_parser_matches_biopython() {
        // Compare id and sequence length for every record against
        // BioPython's SeqIO.parse output on the same file.
        let reference = include_str!("../tests/test_data/fasta_reference.tsv");
        let fasta_data = include_bytes!("../tests/test_data/composition.fa");
        let records = parse_fasta(&fasta_data[..]);

        let ref_entries: Vec<(&str, usize)> = reference
            .lines()
            .map(|line| {
                let mut parts = line.split('\t');
                let id = parts.next().unwrap();
                let len: usize = parts.next().unwrap().parse().unwrap();
                (id, len)
            })
            .collect();

        assert_eq!(
            records.len(),
            ref_entries.len(),
            "record count mismatch: Rust={} Python={}",
            records.len(),
            ref_entries.len()
        );

        for (record, (expected_id, expected_len)) in records.iter().zip(ref_entries.iter()) {
            assert_eq!(
                &record.id, expected_id,
                "id mismatch at record {expected_id}"
            );
            assert_eq!(
                record.seq.len(),
                *expected_len,
                "length mismatch for {expected_id}: Rust={} Python={expected_len}",
                record.seq.len()
            );
        }
    }

    #[test]
    fn fasta_parser_edge_cases() {
        let input = b">seq1 some description\nACGT\nacgt\n\nNNNN\n>seq2\n  ACGT  \n\tTTTT\n";
        let records = parse_fasta(&input[..]);

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id, "seq1");
        assert_eq!(records[0].seq, b"ACGTacgtNNNN");
        assert_eq!(records[1].id, "seq2");
        assert_eq!(records[1].seq, b"ACGTTTTT");
    }

    #[test]
    fn load_composition_matches_python() {
        // Call Python load_composition, compare normalized+logged values bit-exact.
        let py_output = std::process::Command::new("python3")
            .arg("-c")
            .arg(r#"
from concoct.input import load_composition
import numpy as np
comp, lengths = load_composition("tests/test_data/composition.fa", 4, 1000)
for contig_id in comp.index:
    row = comp.loc[contig_id].values
    length = int(lengths[contig_id])
    # Output as hex float for bit-exact comparison
    vals = " ".join(float.hex(v) for v in row)
    print(f"{contig_id}\t{length}\t{vals}")
"#)
            .output()
            .expect("failed to run python3");
        assert!(py_output.status.success(), "Python failed: {}",
            String::from_utf8_lossy(&py_output.stderr));
        let py_stdout = String::from_utf8(py_output.stdout).unwrap();

        let fasta_data = include_bytes!("../tests/test_data/composition.fa");
        let result = load_composition(&fasta_data[..], 4, 1000);

        let py_lines: Vec<&str> = py_stdout.lines().collect();
        assert_eq!(result.contig_ids.len(), py_lines.len());

        for (i, py_line) in py_lines.iter().enumerate() {
            let mut parts = py_line.split('\t');
            let py_id = parts.next().unwrap();
            let _py_len: usize = parts.next().unwrap().parse().unwrap();
            let py_vals: Vec<f64> = parts.next().unwrap()
                .split(' ')
                .map(|s| {
                    // Parse hex float: Python's float.hex() format
                    let s = s.trim();
                    if s.starts_with('-') {
                        -f64_from_hex(&s[1..])
                    } else {
                        f64_from_hex(s)
                    }
                })
                .collect();

            assert_eq!(&result.contig_ids[i], py_id);
            let row_start = i * result.n_features;
            let row = &result.data[row_start..row_start + result.n_features];

            for (j, (&r, &p)) in row.iter().zip(py_vals.iter()).enumerate() {
                // Allow ≤1 ULP: numpy uses its own log implementation
                // which can differ from Rust's f64::ln() by 1 ULP.
                // End-to-end output hash is the real equivalence proof.
                let r_bits = r.to_bits();
                let p_bits = p.to_bits();
                let diff = if r_bits > p_bits { r_bits - p_bits } else { p_bits - r_bits };
                assert!(diff <= 1,
                    "contig {py_id} feature {j}: Rust={r} Python={p} ({diff} ULP)");
            }
        }
    }

    /// Parse Python's float.hex() format (e.g. "0x1.999999999999ap-4")
    fn f64_from_hex(s: &str) -> f64 {
        // Format: 0x1.MMMMMMMMMMMMMp±EEE
        let s = s.strip_prefix("0x").unwrap_or(s);
        let (mantissa_str, exp_str) = s.split_once('p').unwrap();
        let exp: i32 = exp_str.parse().unwrap();

        let (int_part, frac_part) = mantissa_str.split_once('.').unwrap_or((mantissa_str, ""));
        let int_val: u64 = u64::from_str_radix(int_part, 16).unwrap();
        let frac_val: f64 = if frac_part.is_empty() {
            0.0
        } else {
            let frac_int = u64::from_str_radix(frac_part, 16).unwrap();
            frac_int as f64 / 16f64.powi(frac_part.len() as i32)
        };

        (int_val as f64 + frac_val) * 2f64.powi(exp)
    }

    #[test]
    fn feature_mapping_kmer2() {
        // For k=2, Python gives 10 canonical features (16 k-mers, 6 pairs of rev-comp).
        let (mapping, nr_features) = generate_feature_mapping(2);
        assert_eq!(nr_features, 10);
        assert_eq!(mapping.len(), 16); // all 16 2-mers present

        // AA and TT are reverse complements → same index
        assert_eq!(mapping[b"AA".as_ref()], mapping[b"TT".as_ref()]);
        // AT is its own reverse complement
        assert_eq!(mapping[b"AT".as_ref()], mapping[b"AT".as_ref()]);
        // GC and GC — also self-complementary
        assert_eq!(mapping[b"GC".as_ref()], mapping[b"GC".as_ref()]);
        // AG and CT are reverse complements
        assert_eq!(mapping[b"AG".as_ref()], mapping[b"CT".as_ref()]);
    }

    #[test]
    fn calculate_composition_matches_python() {
        // Call Python to get reference output, compare with Rust.
        let py_output = std::process::Command::new("python3")
            .arg("-c")
            .arg(r#"
from concoct.input import _calculate_composition
comp, lengths = _calculate_composition("tests/test_data/composition.fa", 1000, 4)
for contig_id in comp.index:
    row = comp.loc[contig_id].values
    length = int(lengths[contig_id])
    vals = " ".join(str(int(v)) for v in row)
    print(f"{contig_id}\t{length}\t{vals}")
"#)
            .output()
            .expect("failed to run python3");
        assert!(py_output.status.success(), "Python failed: {}", String::from_utf8_lossy(&py_output.stderr));
        let py_stdout = String::from_utf8(py_output.stdout).unwrap();

        let fasta_data = include_bytes!("../tests/test_data/composition.fa");
        let rust_result = calculate_composition(&fasta_data[..], 1000, 4);

        let py_lines: Vec<&str> = py_stdout.lines().collect();
        assert_eq!(rust_result.len(), py_lines.len(),
            "contig count mismatch: Rust={} Python={}", rust_result.len(), py_lines.len());

        for (contig, py_line) in rust_result.iter().zip(py_lines.iter()) {
            let mut parts = py_line.split('\t');
            let py_id = parts.next().unwrap();
            let py_len: usize = parts.next().unwrap().parse().unwrap();
            let py_counts: Vec<f64> = parts.next().unwrap()
                .split(' ')
                .map(|s| s.parse::<f64>().unwrap())
                .collect();

            assert_eq!(&contig.id, py_id, "id mismatch");
            assert_eq!(contig.length, py_len, "length mismatch for {py_id}");
            assert_eq!(contig.counts.len(), py_counts.len(),
                "feature count mismatch for {py_id}");

            for (i, (r, p)) in contig.counts.iter().zip(py_counts.iter()).enumerate() {
                assert_eq!(r.to_bits(), p.to_bits(),
                    "feature {i} mismatch for {py_id}: Rust={r} Python={p}");
            }
        }
    }

    #[test]
    fn feature_mapping_kmer4_matches_python() {
        // Compare every k-mer→index pair against the Python reference.
        // Reference generated by: concoct/input.py generate_feature_mapping(4)
        let reference = include_str!("../tests/test_data/feature_mapping_k4.tsv");
        let (mapping, nr_features) = generate_feature_mapping(4);
        assert_eq!(nr_features, 136);

        let mut checked = 0;
        for line in reference.lines() {
            let mut parts = line.split('\t');
            let kmer_str = parts.next().unwrap();
            let expected_idx: usize = parts.next().unwrap().parse().unwrap();

            let kmer_bytes = kmer_str.as_bytes().to_vec();
            let rust_idx = mapping[&kmer_bytes];
            assert_eq!(
                rust_idx, expected_idx,
                "k-mer {kmer_str}: Rust gives {rust_idx}, Python gives {expected_idx}"
            );
            checked += 1;
        }
        assert_eq!(checked, 256, "should check all 256 4-mers");
    }
}
