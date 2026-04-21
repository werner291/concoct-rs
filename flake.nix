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

        # Tests require the full environment; run them via devShell instead
        doCheck = false;
      };
    in
    {
      packages.${system}.default = concoct;

      devShells.${system}.default = pkgs.mkShell {
        packages = [
          (python.withPackages (ps: [
            concoct
            ps.pytest
          ]))
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
