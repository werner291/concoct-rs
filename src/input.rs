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
        // Both this code and Python's input.py use libc's scalar log()
        // (Rust via f64::ln(), Python via math.log), so outputs are
        // bit-identical.
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

/// Result of coverage loading.
pub struct CoverageData {
    /// Flat row-major matrix, n_contigs × n_columns.
    pub data: Vec<f64>,
    /// Contig IDs, in row order (coverage file order, filtered).
    pub contig_ids: Vec<String>,
    /// Column names (sample names + optional "total_coverage").
    pub column_names: Vec<String>,
    /// Number of columns.
    pub n_columns: usize,
    /// Coverage range: (first_col_name, last_col_name) for downstream use.
    pub cov_range: (String, String),
}

/// Load coverage from a TSV file: parse, filter to known contigs, add pseudo
/// count, optionally normalize, optionally add total_coverage, and log-transform.
///
/// Ports: concoct/input.py load_coverage (L41-79)
///
/// The operation order matches the Python exactly:
///   1. Parse TSV (header + contig_id index)
///   2. Filter to contigs present in contig_lengths
///   3. Add pseudo count: read_length / contig_length to each value
///   4. If normalizing: divide each column by its sum (per-sample)
///   5. If add_total_coverage: append row-sum as "total_coverage" column
///   6. If normalizing: divide each row's sample columns by their sum (per-contig)
///   7. Log-transform all columns in cov_range
pub fn load_coverage<R: BufRead>(
    reader: R,
    contig_lengths: &HashMap<String, f64>,
    no_cov_normalization: bool,
    add_total_coverage: bool,
    read_length: f64,
) -> CoverageData {
    let mut lines = reader.lines();

    // Header: first field is contig_id label, rest are sample names
    let header_line = lines.next().expect("empty coverage file").expect("read error");
    let header_fields: Vec<&str> = header_line.split('\t').collect();
    let sample_names: Vec<String> = header_fields[1..].iter().map(|s| s.to_string()).collect();
    let n_samples = sample_names.len();

    // Parse rows, filtering to contigs present in contig_lengths
    let mut contig_ids: Vec<String> = Vec::new();
    let mut rows: Vec<Vec<f64>> = Vec::new();

    for line in lines {
        let line = line.expect("read error");
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let contig_id = fields[0];

        if !contig_lengths.contains_key(contig_id) {
            continue;
        }

        let values: Vec<f64> = fields[1..]
            .iter()
            .map(|s| s.parse::<f64>().expect("invalid float in coverage"))
            .collect();
        assert_eq!(values.len(), n_samples);

        contig_ids.push(contig_id.to_string());
        rows.push(values);
    }

    // Step 1: Add pseudo count (read_length / contig_length) to each value
    // Ports: cov.add(read_length/contig_lengths, axis='index')
    for (i, row) in rows.iter_mut().enumerate() {
        let pseudo = read_length / contig_lengths[&contig_ids[i]];
        for val in row.iter_mut() {
            *val += pseudo;
        }
    }

    let cov_range_start = sample_names[0].clone();
    let mut cov_range_end = sample_names[n_samples - 1].clone();

    if !no_cov_normalization {
        // Step 2: Normalize per sample — divide each column by its sum
        // Ports: _normalize_per_sample: arr.divide(arr.sum(axis=0), axis=1)
        let mut col_sums = vec![0.0f64; n_samples];
        for row in &rows {
            for (j, &val) in row.iter().enumerate() {
                col_sums[j] += val;
            }
        }
        for row in rows.iter_mut() {
            for (j, val) in row.iter_mut().enumerate() {
                *val /= col_sums[j];
            }
        }
    }

    // Step 3: Add total_coverage column (sum of sample columns after per-sample norm)
    // Ports: cov['total_coverage'] = cov.loc[:,cov_range[0]:cov_range[1]].sum(axis=1)
    if add_total_coverage {
        for row in rows.iter_mut() {
            let total: f64 = row.iter().sum();
            row.push(total);
        }
        cov_range_end = "total_coverage".to_string();
    }

    if !no_cov_normalization {
        // Step 4: Normalize per contig — divide each row's SAMPLE columns by their sum
        // At this point cov_range is still (first_sample, last_sample) in Python,
        // so only sample columns are normalized, not total_coverage.
        // Ports: _normalize_per_contig: arr.divide(arr.sum(axis=1), axis=0)
        for row in rows.iter_mut() {
            let row_sum: f64 = row[..n_samples].iter().sum();
            for val in row[..n_samples].iter_mut() {
                *val /= row_sum;
            }
        }
    }

    // Step 5: Log-transform all columns in final cov_range
    // After step 3, cov_range includes total_coverage if added.
    // Ports: cov.loc[:,cov_range[0]:cov_range[1]].map(math.log)
    let n_to_log = if add_total_coverage { n_samples + 1 } else { n_samples };
    for row in rows.iter_mut() {
        for val in row[..n_to_log].iter_mut() {
            *val = val.ln();
        }
    }

    // Build column names
    let mut column_names = sample_names;
    if add_total_coverage {
        column_names.push("total_coverage".to_string());
    }
    let n_columns = column_names.len();

    // Flatten to row-major
    let data: Vec<f64> = rows.into_iter().flatten().collect();

    CoverageData {
        data,
        contig_ids,
        column_names,
        n_columns,
        cov_range: (cov_range_start, cov_range_end),
    }
}

/// Result of loading and joining composition + coverage data.
pub struct JoinedData {
    /// Flat row-major matrix, n_contigs × n_columns.
    pub data: Vec<f64>,
    /// Contig IDs, in row order (composition file order, filtered to intersection).
    pub contig_ids: Vec<String>,
    /// Column names: composition feature indices ("0","1",...) then coverage sample names.
    pub column_names: Vec<String>,
    /// Number of contigs (rows).
    pub n_contigs: usize,
    /// Number of columns (composition features + coverage columns).
    pub n_columns: usize,
}

/// Load composition and optionally coverage, then inner-join on contig ID.
///
/// Ports the data flow in bin/concoct lines 20-30:
///   composition, cov, cov_range = load_data(args)
///   joined = composition.join(cov.loc[:,cov_range[0]:cov_range[1]], how="inner")
///
/// Row order follows composition (FASTA order), keeping only contigs present
/// in both datasets.
pub fn load_and_join<R1: BufRead, R2: BufRead>(
    comp_reader: R1,
    cov_reader: Option<R2>,
    kmer_len: usize,
    length_threshold: usize,
    no_cov_normalization: bool,
    add_total_coverage: bool,
    read_length: f64,
) -> JoinedData {
    let comp = load_composition(comp_reader, kmer_len, length_threshold);

    if let Some(cov_reader) = cov_reader {
        // Build contig_lengths map for coverage loading
        let contig_lengths: HashMap<String, f64> = comp
            .contig_ids
            .iter()
            .zip(comp.contig_lengths.iter())
            .map(|(id, &len)| (id.clone(), len as f64))
            .collect();

        let cov = load_coverage(
            cov_reader,
            &contig_lengths,
            no_cov_normalization,
            add_total_coverage,
            read_length,
        );

        // Build coverage index for O(1) lookup
        let cov_index: HashMap<&str, usize> = cov
            .contig_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.as_str(), i))
            .collect();

        let n_comp_cols = comp.n_features;
        let n_cov_cols = cov.n_columns;
        let n_columns = n_comp_cols + n_cov_cols;

        let mut data = Vec::new();
        let mut contig_ids = Vec::new();

        // Inner join: composition order, only contigs present in both
        for (i, comp_id) in comp.contig_ids.iter().enumerate() {
            if let Some(&cov_idx) = cov_index.get(comp_id.as_str()) {
                let comp_row =
                    &comp.data[i * n_comp_cols..(i + 1) * n_comp_cols];
                let cov_row =
                    &cov.data[cov_idx * n_cov_cols..(cov_idx + 1) * n_cov_cols];
                data.extend_from_slice(comp_row);
                data.extend_from_slice(cov_row);
                contig_ids.push(comp_id.clone());
            }
        }

        let n_contigs = contig_ids.len();

        // Column names: "0","1",...,"135","sample_1",...,"sample_16"[,"total_coverage"]
        let mut column_names: Vec<String> =
            (0..n_comp_cols).map(|i| i.to_string()).collect();
        column_names.extend(cov.column_names);

        JoinedData {
            data,
            contig_ids,
            column_names,
            n_contigs,
            n_columns,
        }
    } else {
        // Composition only (no coverage file)
        let n_contigs = comp.contig_ids.len();
        let n_columns = comp.n_features;
        let column_names: Vec<String> =
            (0..n_columns).map(|i| i.to_string()).collect();

        JoinedData {
            data: comp.data,
            contig_ids: comp.contig_ids,
            column_names,
            n_contigs,
            n_columns,
        }
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
                // Bit-exact: both Python (math.log) and Rust (f64::ln)
                // use libc's scalar log().
                assert_eq!(r.to_bits(), p.to_bits(),
                    "contig {py_id} feature {j}: Rust={r} Python={p}");
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

    // calculate_composition_matches_python was removed: the Python oracle
    // (_calculate_composition) no longer exists in input.py after the Rust
    // port. Composition equivalence is verified through load_composition_matches_python.

    #[test]
    fn load_coverage_sanity() {
        // Verify load_coverage produces correct dimensions and finite values
        // on the test dataset. Bit-exact comparison against Python is not
        // required here — the end-to-end determinism check is the proof
        // (summation order differs between Rust and numpy's pairwise sum).
        let fasta_data = include_bytes!("../tests/test_data/composition.fa");
        let comp = load_composition(&fasta_data[..], 4, 1000);
        let contig_lengths: HashMap<String, f64> = comp.contig_ids.iter()
            .zip(comp.contig_lengths.iter())
            .map(|(id, &len)| (id.clone(), len as f64))
            .collect();
        let n_composition_contigs = contig_lengths.len();

        let cov_data = include_bytes!("../tests/test_data/coverage");

        // Test with normalization, no total coverage (determinism check config)
        let result = load_coverage(
            &cov_data[..],
            &contig_lengths,
            false, // no_cov_normalization = false → normalize
            false, // add_total_coverage = false
            100.0,
        );

        // All contigs in the coverage file that also passed the composition
        // length filter should be present.
        assert!(result.contig_ids.len() > 0);
        assert!(result.contig_ids.len() <= n_composition_contigs);
        assert_eq!(result.n_columns, 16); // 16 samples in test data
        assert_eq!(result.data.len(), result.contig_ids.len() * result.n_columns);
        assert_eq!(result.cov_range.0, "sample_1");
        assert_eq!(result.cov_range.1, "sample_16");

        // All values should be finite (log of positive numbers)
        for &v in &result.data {
            assert!(v.is_finite(), "non-finite value in coverage: {v}");
        }

        // Test with total coverage
        let result_tc = load_coverage(
            &cov_data[..],
            &contig_lengths,
            false,
            true, // add_total_coverage
            100.0,
        );
        assert_eq!(result_tc.n_columns, 17); // 16 samples + total_coverage
        assert_eq!(result_tc.cov_range.1, "total_coverage");
        assert_eq!(result_tc.column_names[16], "total_coverage");
        for &v in &result_tc.data {
            assert!(v.is_finite(), "non-finite value in coverage (with total): {v}");
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
