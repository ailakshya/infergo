use std::env;
use std::path::PathBuf;

fn main() {
    // Search paths for libinfer_api:
    // 1. INFERGO_LIB_DIR environment variable (highest priority)
    // 2. ../../../build (relative to this crate, i.e. the project build dir)
    // 3. /usr/local/lib (system default)

    let mut search_paths: Vec<PathBuf> = Vec::new();

    if let Ok(lib_dir) = env::var("INFERGO_LIB_DIR") {
        search_paths.push(PathBuf::from(lib_dir));
    }

    // Project build directory (cgo/build)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_build = manifest_dir.join("../../../build");
    if project_build.exists() {
        search_paths.push(project_build.canonicalize().unwrap());
    }

    search_paths.push(PathBuf::from("/usr/local/lib"));

    for path in &search_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    println!("cargo:rustc-link-lib=dylib=infer_api");

    // Re-run if the env var changes
    println!("cargo:rerun-if-env-changed=INFERGO_LIB_DIR");

    // Include path for the C header (optional, for bindgen users)
    let include_dir = manifest_dir.join("../../../cpp/include");
    if include_dir.exists() {
        println!(
            "cargo:include={}",
            include_dir.canonicalize().unwrap().display()
        );
    }
}
