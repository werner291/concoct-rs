{
  description = "CONCOCT - Clustering cONtigs with COverage and ComposiTion";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    crane.url = "github:ipetkov/crane";
  };

  outputs = { self, nixpkgs, crane }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python3;

      craneLib = crane.mkLib pkgs;

      bcbio-gff = pkgs.callPackage ./nix/bcbio-gff.nix {
        inherit (python.pkgs) buildPythonPackage fetchPypi setuptools biopython six;
      };

      concoct = python.pkgs.buildPythonPackage {
        pname = "concoct";
        version = "1.1.0";
        src = ./.;
        format = "setuptools";

        propagatedBuildInputs = with python.pkgs; [
          numpy
          setuptools
        ];

        # The vbgmm extension is now provided by the Rust PyO3 module
        # instead of the old Cython/C build.
        postInstall = ''
          cp ${pyo3Module}/lib/*.so $out/${python.sitePackages}/
        '';

        doCheck = false;
      };

      testPython = python.withPackages (ps: [
        concoct
        bcbio-gff
        ps.biopython
        ps.pandas
        ps.pytest
      ]);

      integrationTestData = pkgs.fetchzip {
        url = "https://github.com/BinPro/integration_test_data/archive/v1.0.tar.gz";
        hash = "sha256-nCecnv+eIqyCzD9v56dXur3QNBgW7RRa6BkNVRHcQFE=";
      };

      mkPytestCheck = name: testPath: { extraPackages ? [], needsIntegrationData ? false }: pkgs.runCommand "check-${name}" {
        nativeBuildInputs = [ testPython ] ++ extraPackages;
      } (''
        cp -r ${./.}/tests ${./.}/scripts ${./.}/scgs $TMPDIR/
        chmod -R u+w $TMPDIR/tests
        cd $TMPDIR
      '' + pkgs.lib.optionalString needsIntegrationData ''
        ln -s ${integrationTestData} tests/test_data/integration_test_data
      '' + ''
        python3 -m pytest ${testPath} -v --tb=short
        touch $out
      '');
      dockerImage = pkgs.callPackage ./nix/docker-image.nix {
        inherit (pkgs) dockerTools bashInteractive coreutils samtools bedtools perl;
        inherit testPython;
      };

      mkDeterminismCheck = name: covFile: compFile: clusters:
        pkgs.callPackage ./nix/determinism-check.nix {
          inherit testPython name covFile compFile clusters;
          src = ./.;
        };

      # Crane: Rust crate with C FFI to the VBGMM library
      rustSrc = pkgs.lib.cleanSourceWith {
        src = ./.;
        filter = path: type:
          (craneLib.filterCargoSources path type)
          || (builtins.match ".*c-concoct/.*" path != null)
          || (builtins.match ".*tests/test_data/.*" path != null);
      };

      commonArgs = {
        src = rustSrc;
        nativeBuildInputs = [ pkgs.pkg-config ];
        buildInputs = [ pkgs.gsl pkgs.lapack ];
        # Pin target-cpu to baseline to avoid AVX float divergence
        RUSTFLAGS = "-C target-cpu=x86-64";
      };

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      concoctRust = craneLib.buildPackage (commonArgs // {
        inherit cargoArtifacts;
        # Tests are covered by concoctRustTests (which has testPython
        # for oracle comparisons). This derivation only builds the binary.
        doCheck = false;
      });

      concoctRustTests = craneLib.cargoTest (commonArgs // {
        inherit cargoArtifacts;
        # Tests that compare against Python oracle need the concoct package.
        nativeBuildInputs = (commonArgs.nativeBuildInputs or []) ++ [ testPython ];
      });

      concoctRustBench = craneLib.mkCargoDerivation (commonArgs // {
        inherit cargoArtifacts;
        pnameSuffix = "-bench";
        buildPhaseCargoCommand = "cargo bench --bench bench_leaf";
        installPhaseCommand = "mkdir -p $out";
      });

      camiDataset = import ./nix/cami-dataset.nix {
        inherit (pkgs) fetchurl runCommand;
      };

      pyo3Module = import ./nix/pyo3-module.nix {
        inherit (pkgs) lib;
        inherit craneLib rustSrc commonArgs;
        python3 = python;
      };

    in
    {
      packages.${system} = {
        default = concoct;
        pyo3 = pyo3Module;
        docker = dockerImage;
        bench = concoctRustBench;
        codegen-calcdist = pkgs.callPackage ./nix/codegen-calcdist.nix {
          inherit (pkgs) gcc gsl;
          inherit craneLib rustSrc commonArgs cargoArtifacts;
        };
      };

      checks.${system} = {
        rust-tests = concoctRustTests;

        pytest-unit-input = mkPytestCheck "unit-input"
          "tests/test_unittest_input.py" {};
        pytest-cut-up-fasta = mkPytestCheck "cut-up-fasta"
          "tests/test_cut_up_fasta.py" {};
        pytest-gen-input-table = mkPytestCheck "gen-input-table"
          "tests/test_gen_input_table.py" { extraPackages = [ pkgs.bedtools ]; };
        pytest-integration = mkPytestCheck "integration"
          "tests/test_integration.py" { extraPackages = [ pkgs.perl ]; };
        pytest-cog-table = mkPytestCheck "cog-table"
          "tests/test_COG_table.py" {};
        pytest-merge-cutup = mkPytestCheck "merge-cutup"
          "tests/test_merge_cutup_clustering.py" { needsIntegrationData = true; };
        pytest-integration-scripts = mkPytestCheck "integration-scripts"
          "tests/test_integration_with_scripts.py" { needsIntegrationData = true; extraPackages = [ pkgs.samtools pkgs.bedtools pkgs.perl ]; };

        # Verify CONCOCT produces identical output across repeated runs
        # and across thread counts (1 vs 4), given the same seed.
        determinism-small = mkDeterminismCheck "small"
          "test_data/coverage"
          "test_data/composition.fa"
          10;
        determinism-large = mkDeterminismCheck "large"
          "test_data/large_contigs/coverage_table.tsv"
          "test_data/large_contigs/contigs.fa"
          20;

        # Verify the Rust binary produces identical output to the Python pipeline.
        rust-binary-small = import ./nix/rust-binary-check.nix {
          inherit (pkgs) runCommand;
          inherit concoctRust;
          src = ./.;
          name = "small";
          covFile = "test_data/coverage";
          compFile = "test_data/composition.fa";
          clusters = 10;
          expectedHash = "3bdec94c13d8bba9c0d778381956697f253d178d2338e14ba1e0c07535969058";
        };
        rust-binary-large = import ./nix/rust-binary-check.nix {
          inherit (pkgs) runCommand;
          inherit concoctRust;
          src = ./.;
          name = "large";
          covFile = "test_data/large_contigs/coverage_table.tsv";
          compFile = "test_data/large_contigs/contigs.fa";
          clusters = 20;
          expectedHash = "9ff1e1f81fa5a593db000da1a8c052c089b3e5b944972048b969d726c92373d2";
        };

        # CAMI I High: 39k contigs × 5 samples. Realistic scale test.
        # Runs both the Python pipeline and the Rust binary, then
        # compares their output hashes.
        # Run explicitly: nix build .#checks.x86_64-linux.equivalence-cami
        equivalence-cami = pkgs.runCommand "check-equivalence-cami" {
          nativeBuildInputs = [ testPython concoctRust pkgs.gzip ];
        } ''
          gzip -dc ${camiDataset.contigs} > $TMPDIR/contigs.fa

          # Python pipeline (uses PyO3 module under the hood)
          python_dir=$TMPDIR/python_out
          python3 -c "
          import sys; sys.argv = ['concoct',
            '--coverage_file', '${camiDataset.coverage}',
            '--composition_file', '$TMPDIR/contigs.fa',
            '--basename', '$python_dir/',
            '-c', '400', '--no_total_coverage', '--seed', '1', '--threads', '1']
          exec(open('${./.}/bin/concoct').read())
          " 2>/dev/null
          py_hash=$(sha256sum $python_dir/clustering_gt1000.csv | cut -d' ' -f1)
          echo "Python hash: $py_hash"

          # Rust binary
          rust_dir=$TMPDIR/rust_out
          concoct --coverage_file ${camiDataset.coverage} \
                  --composition_file $TMPDIR/contigs.fa \
                  --basename $rust_dir/ \
                  -c 400 --no_total_coverage --seed 1 --threads 1 2>/dev/null
          rs_hash=$(sha256sum $rust_dir/clustering_gt1000.csv | cut -d' ' -f1)
          echo "Rust hash:   $rs_hash"

          if [ "$py_hash" != "$rs_hash" ]; then
            echo "MISMATCH: Python and Rust produced different output on CAMI I High"
            diff <(sort $python_dir/clustering_gt1000.csv) <(sort $rust_dir/clustering_gt1000.csv) | head -20
            exit 1
          fi

          # Also verify Rust is deterministic across thread counts
          rust_dir4=$TMPDIR/rust_out_t4
          concoct --coverage_file ${camiDataset.coverage} \
                  --composition_file $TMPDIR/contigs.fa \
                  --basename $rust_dir4/ \
                  -c 400 --no_total_coverage --seed 1 --threads 4 2>/dev/null
          rs4_hash=$(sha256sum $rust_dir4/clustering_gt1000.csv | cut -d' ' -f1)
          echo "Rust t=4:    $rs4_hash"

          if [ "$rs_hash" != "$rs4_hash" ]; then
            echo "MISMATCH: Rust binary not deterministic across thread counts"
            exit 1
          fi

          echo "Python and Rust produce identical output on CAMI I High: $py_hash"
          touch $out
        '';

        docker = import ./nix/docker-test.nix {
          inherit pkgs dockerImage;
          src = ./.;
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          testPython
          pkgs.gsl
          pkgs.lapack
          pkgs.gcc
          pkgs.bedtools
          pkgs.samtools
          pkgs.perl
          pkgs.rustc
          pkgs.cargo
          pkgs.pkg-config
          pkgs.gh
        ];

        shellHook = ''
          export C_INCLUDE_PATH="${pkgs.gsl}/include:''${C_INCLUDE_PATH:-}"
          export LIBRARY_PATH="${pkgs.gsl}/lib:${pkgs.lapack}/lib:''${LIBRARY_PATH:-}"
          export LD_LIBRARY_PATH="${pkgs.gsl}/lib:${pkgs.lapack}/lib:''${LD_LIBRARY_PATH:-}"
          if [ ! -e tests/test_data/integration_test_data ]; then
            ln -s ${integrationTestData} tests/test_data/integration_test_data
          fi

          # Install pre-commit hook that runs nix flake check
          if [ -d .git ]; then
            mkdir -p .git/hooks
            cat > .git/hooks/pre-commit << 'HOOK'
#!/usr/bin/env bash
echo "Running nix flake check before commit..."
nix flake check
HOOK
            chmod +x .git/hooks/pre-commit
          fi
        '';
      };
    };
}
