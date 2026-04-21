{
  description = "CONCOCT - Clustering cONtigs with COverage and ComposiTion";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      python = pkgs.python3;

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
    in
    {
      packages.${system}.default = concoct;

      checks.${system} = {
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
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          testPython
          pkgs.gsl
          pkgs.gcc
          pkgs.bedtools
          pkgs.samtools
          pkgs.perl
        ];

        shellHook = ''
          export C_INCLUDE_PATH="${pkgs.gsl}/include:''${C_INCLUDE_PATH:-}"
          export LIBRARY_PATH="${pkgs.gsl}/lib:''${LIBRARY_PATH:-}"
          export LD_LIBRARY_PATH="${pkgs.gsl}/lib:''${LD_LIBRARY_PATH:-}"
          if [ ! -e tests/test_data/integration_test_data ]; then
            ln -s ${integrationTestData} tests/test_data/integration_test_data
          fi
        '';
      };
    };
}
