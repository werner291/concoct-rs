{ runCommand, gcc, gsl, craneLib, rustSrc, commonArgs, cargoArtifacts }:

let
  # Build the crate with --emit asm
  rustAsm = craneLib.mkCargoDerivation (commonArgs // {
    inherit cargoArtifacts;
    pnameSuffix = "-asm";
    buildPhaseCargoCommand = "cargo rustc --lib --release -- --emit asm";
    installPhaseCommand = ''
      mkdir -p $out
      find target/release/deps -name "concoct-*.s" -exec cp {} $out/calcdist_rust.s \;
      sed -n '/_ZN7concoct5vbgmm9calc_dist/,/\.cfi_endproc/p' $out/calcdist_rust.s > $out/calcdist_rust_func.s
    '';
  });
in
runCommand "codegen-calcdist" {
  nativeBuildInputs = [ gcc ];
  buildInputs = [ gsl ];
} ''
  mkdir -p $out

  # --- C codegen ---
  gcc -O3 -std=c99 -fno-strict-overflow -S -o $out/calcdist_c.s \
    -I ${gsl.dev}/include \
    -x c - << 'CCODE'
#include <math.h>
double calcDist(double* adX, double *adMu, int nD)
{
  double dDist = 0.0;
  int i = 0;
  for(i = 0; i < nD; i++){
    double dV = adX[i] - adMu[i];
    dDist += dV*dV;
  }
  return sqrt(dDist);
}
CCODE
  sed -n '/^calcDist:/,/\.cfi_endproc/p' $out/calcdist_c.s > $out/calcdist_c_func.s

  # --- Rust codegen (from crane build) ---
  cp ${rustAsm}/calcdist_rust_func.s $out/calcdist_rust_func.s
  cp ${rustAsm}/calcdist_rust.s $out/calcdist_rust.s

  echo "=== C (GCC $(gcc --version | head -1)) ==="
  cat $out/calcdist_c_func.s
  echo ""
  echo "=== Rust (LLVM via crane) ==="
  cat $out/calcdist_rust_func.s
''
