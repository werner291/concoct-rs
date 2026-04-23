fn main() {
    // Build c_vbgmm_fit.c as a static library for FFI oracle testing.
    // Flags match the nix Python build exactly: -O3 -std=c99 -fopenmp
    // No -ffast-math — IEEE 754 semantics are required.
    cc::Build::new()
        .file("c-concoct/c_vbgmm_fit.c")
        .file("c-concoct/ffi_wrappers.c")
        .flag("-O3")
        .flag("-std=c99")
        .flag("-fopenmp")
        .flag("-fno-strict-overflow")
        .include("c-concoct")
        .warnings(false)
        .compile("c_vbgmm_fit");

    println!("cargo:rustc-link-lib=gsl");
    println!("cargo:rustc-link-lib=gslcblas");
    println!("cargo:rustc-link-lib=gomp");
    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rerun-if-changed=c-concoct/c_vbgmm_fit.c");
    println!("cargo:rerun-if-changed=c-concoct/c_vbgmm_fit.h");
    println!("cargo:rerun-if-changed=c-concoct/ffi_wrappers.c");
}
