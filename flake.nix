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

        nativeBuildInputs = [ python.pkgs.cython ];

        buildInputs = [ pkgs.gsl ];

        propagatedBuildInputs = with python.pkgs; [
          numpy
          scipy
          pandas
          biopython
          scikit-learn
          setuptools
        ];

        doCheck = false;
      };

      testPython = python.withPackages (ps: [
        concoct
        bcbio-gff
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
          || (builtins.match ".*c-concoct/.*" path != null);
      };

      commonArgs = {
        src = rustSrc;
        nativeBuildInputs = [ pkgs.pkg-config ];
        buildInputs = [ pkgs.gsl ];
        # Pin target-cpu to baseline to avoid AVX float divergence
        RUSTFLAGS = "-C target-cpu=x86-64";
      };

      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      concoctRust = craneLib.buildPackage (commonArgs // {
        inherit cargoArtifacts;
      });

      concoctRustTests = craneLib.cargoTest (commonArgs // {
        inherit cargoArtifacts;
      });

      concoctRustBench = craneLib.mkCargoDerivation (commonArgs // {
        inherit cargoArtifacts;
        pnameSuffix = "-bench";
        buildPhaseCargoCommand = "cargo bench --bench bench_leaf";
        installPhaseCommand = "mkdir -p $out";
      });

      pyo3Module = import ./nix/pyo3-module.nix {
        inherit (pkgs) lib;
        inherit craneLib rustSrc commonArgs;
        python3 = python;
      };

      # Python with both the old Cython module and access to the new PyO3 one
      testPythonPyo3 = python.withPackages (ps: [
        concoct
        ps.numpy
        ps.pytest
      ]);
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

        # Verify PyO3 vbgmm.fit produces identical output to Cython
        pyo3-equivalence = pkgs.runCommand "check-pyo3-equivalence" {
          nativeBuildInputs = [ testPythonPyo3 ];
        } ''
          export VBGMM_PYO3_LIB="${pyo3Module}/lib"
          python3 -m pytest ${./.}/tests/test_pyo3_equivalence.py -v --tb=short
          touch $out
        '';

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

        docker = import ./nix/docker-test.nix {
          inherit pkgs dockerImage;
          src = ./.;
        };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          testPython
          pkgs.gsl
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
          export LIBRARY_PATH="${pkgs.gsl}/lib:''${LIBRARY_PATH:-}"
          export LD_LIBRARY_PATH="${pkgs.gsl}/lib:''${LD_LIBRARY_PATH:-}"
          if [ ! -e tests/test_data/integration_test_data ]; then
            ln -s ${integrationTestData} tests/test_data/integration_test_data
          fi

          # Install pre-push hook that runs nix flake check
          if [ -d .git ]; then
            mkdir -p .git/hooks
            cat > .git/hooks/pre-push << 'HOOK'
#!/usr/bin/env bash
echo "Running nix flake check before push..."
nix flake check
HOOK
            chmod +x .git/hooks/pre-push
          fi
        '';
      };
    };
}
