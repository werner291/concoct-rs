# Build the PyO3 vbgmm module as a Python-importable shared library.
#
# Uses crane to build the cdylib, then renames it to the Python extension
# naming convention (vbgmm.cpython-3XX-x86_64-linux-gnu.so).
{ lib
, craneLib
, rustSrc
, commonArgs
, python3
}:

let
  extSuffix = python3.sourceVersion.major + python3.sourceVersion.minor;
  soName = "vbgmm.cpython-${extSuffix}-x86_64-linux-gnu.so";

  # PyO3 needs Python at dep-compilation time, so we build separate
  # cargo artifacts with Python available.
  pyo3CommonArgs = commonArgs // {
    src = rustSrc;
    nativeBuildInputs = (commonArgs.nativeBuildInputs or []) ++ [
      python3
    ];
    PYO3_PYTHON = "${python3}/bin/python3";
    cargoExtraArgs = "--features python";
  };

  pyo3CargoArtifacts = craneLib.buildDepsOnly pyo3CommonArgs;

  cdylib = craneLib.buildPackage (pyo3CommonArgs // {
    cargoArtifacts = pyo3CargoArtifacts;
    pnameSuffix = "-pyo3";

    installPhaseCommand = ''
      mkdir -p $out/lib
      cp target/release/libconcoct.so $out/lib/${soName}
    '';
  });
in
cdylib
