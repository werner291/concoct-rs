{ runCommand, testPython, src, name, covFile, compFile, clusters }:

runCommand "check-determinism-${name}" {
  nativeBuildInputs = [ testPython ];
} ''
  cp -r ${src}/tests $TMPDIR/
  chmod -R u+w $TMPDIR/tests
  cd $TMPDIR/tests

  EXPECTED=""
  for threads in 1 4; do
    for run in 1 2; do
      dir=$TMPDIR/t''${threads}_r''${run}
      concoct --coverage_file ${covFile} \
              --composition_file ${compFile} \
              --basename $dir/ \
              -c ${toString clusters} --no_total_coverage --seed 1 --threads $threads 2>/dev/null
      hash=$(sha256sum $dir/clustering_gt1000.csv | cut -d' ' -f1)
      echo "threads=$threads run=$run hash=$hash"
      if [ -z "$EXPECTED" ]; then
        EXPECTED=$hash
      elif [ "$hash" != "$EXPECTED" ]; then
        echo "MISMATCH: expected $EXPECTED got $hash"
        exit 1
      fi
    done
  done
  echo "All runs identical: $EXPECTED"
  touch $out
''
