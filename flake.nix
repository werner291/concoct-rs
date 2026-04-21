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
        ps.pytest
      ]);

      mkPytestCheck = name: testPath: { extraPackages ? [] }: pkgs.runCommand "check-${name}" {
        nativeBuildInputs = [ testPython ] ++ extraPackages;
      } ''
        cp -r ${./.}/tests ${./.}/scripts $TMPDIR/
        chmod -R u+w $TMPDIR/tests
        cd $TMPDIR
        python3 -m pytest ${testPath} -v --tb=short
        touch $out
      '';
    in
    {
      packages.${system}.default = concoct;

      checks.${system} = {
        pytest-unit-input = mkPytestCheck "unit-input"
          "tests/test_unittest_input.py" {};
        pytest-cut-up-fasta = mkPytestCheck "cut-up-fasta"
          "tests/test_cut_up_fasta.py" {};
        pytest-gen-input-table-bed = mkPytestCheck "gen-input-table-bed"
          "tests/test_gen_input_table.py::TestCMD::test_with_bedfiles" {};
        pytest-integration = mkPytestCheck "integration"
          "tests/test_integration.py" { extraPackages = [ pkgs.perl ]; };
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          testPython
          pkgs.gsl
          pkgs.gcc
        ];

        shellHook = ''
          export C_INCLUDE_PATH="${pkgs.gsl}/include:''${C_INCLUDE_PATH:-}"
          export LIBRARY_PATH="${pkgs.gsl}/lib:''${LIBRARY_PATH:-}"
          export LD_LIBRARY_PATH="${pkgs.gsl}/lib:''${LD_LIBRARY_PATH:-}"
        '';
      };
    };
}
