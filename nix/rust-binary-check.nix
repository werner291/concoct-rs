# Verify the Rust binary produces hash-identical output to the Python pipeline.
#
# Runs the Rust binary on the same test data with the same parameters as the
# determinism checks, and asserts the clustering hash matches.
{ runCommand, concoctRust, src, name, covFile, compFile, clusters, expectedHash }:

runCommand "check-rust-binary-${name}" {
  nativeBuildInputs = [ concoctRust ];
} ''
  cp -r ${src}/tests $TMPDIR/
  chmod -R u+w $TMPDIR/tests
  cd $TMPDIR/tests

  for threads in 1 4; do
    dir=$TMPDIR/t''${threads}
    concoct --coverage_file ${covFile} \
            --composition_file ${compFile} \
            --basename $dir/ \
            -c ${toString clusters} --no_total_coverage --seed 1 --threads $threads 2>/dev/null
    hash=$(sha256sum $dir/clustering_gt1000.csv | cut -d' ' -f1)
    echo "threads=$threads hash=$hash"
    if [ "$hash" != "${expectedHash}" ]; then
      echo "MISMATCH: expected ${expectedHash} got $hash"
      exit 1
    fi
  done
  echo "Rust binary matches expected hash: ${expectedHash}"
  touch $out
''
