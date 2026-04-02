//! Build script: compiles Lua 5.4 from source, finds glslc for shaders.

fn main() {
    build_lua();
    compile_shaders();
}

/// Search for Lua source files across several common layouts.
///
/// Supports:
/// - `arch/lua54/src/*.c`  (lua.org tarball)
/// - `arch/lua54/*.c`      (GitHub lua/lua clone)
/// - `arch/Lua54/src/*.c`  (case variants)
/// - `arch/Lua54/*.c`
fn build_lua() {
    // All candidate directories where Lua .c files might live.
    let candidates = [
        "arch/lua54/src",
        "arch/lua54",
        "arch/Lua54/src",
        "arch/Lua54",
        "arch/lua-5.4/src",
        "arch/lua5.4/src",
    ];

    let lua_dir = candidates.iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.join("lapi.c").exists());

    let lua_dir = match lua_dir {
        Some(d) => {
            println!("cargo:warning=Found Lua sources at: {}", d.display());
            d
        }
        None => {
            println!("cargo:warning=");
            println!("cargo:warning=Lua 5.4 source not found! Tried:");
            for c in &candidates {
                println!("cargo:warning=  {c}/lapi.c");
            }
            println!("cargo:warning=");
            println!("cargo:warning=Fix: clone Lua into arch/lua54/");
            println!("cargo:warning=  git clone https://github.com/lua/lua.git arch/lua54 --branch v5.4.7 --depth 1");
            println!("cargo:warning=");
            return;
        }
    };

    // Rerun if any Lua source changes.
    println!("cargo:rerun-if-changed={}", lua_dir.display());

    // All core + lib sources (excluding lua.c and luac.c / they have main()).
    let core_sources = [
        "lapi.c", "lauxlib.c", "lbaselib.c", "lcode.c", "lcorolib.c",
        "lctype.c", "ldblib.c", "ldebug.c", "ldo.c", "ldump.c",
        "lfunc.c", "lgc.c", "linit.c", "liolib.c", "llex.c",
        "lmathlib.c", "lmem.c", "loadlib.c", "lobject.c", "lopcodes.c",
        "loslib.c", "lparser.c", "lstate.c", "lstring.c", "lstrlib.c",
        "ltable.c", "ltablib.c", "ltm.c", "lundump.c", "lutf8lib.c",
        "lvm.c", "lzio.c",
    ];

    let mut build = cc::Build::new();
    build
        .include(&lua_dir)
        .warnings(false)
        .opt_level(2);

    // Platform-specific defines.
    // On Windows/MSVC: no special defines needed (vanilla Lua works).
    // On Linux/macOS: enable POSIX features.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    match target_os.as_str() {
        "linux" | "freebsd" | "openbsd" | "netbsd" => {
            build.define("LUA_USE_POSIX", None);
            build.define("LUA_USE_DLOPEN", None);
        }
        "macos" => {
            build.define("LUA_USE_MACOSX", None);
        }
        "windows" => {
            // MSVC: nothing extra needed.
            // MinGW: could add LUA_USE_POSIX but let's keep it simple.
            if target_env == "gnu" {
                build.define("LUA_USE_POSIX", None);
            }
        }
        _ => {}
    }

    let mut found = 0;
    for src in &core_sources {
        let path = lua_dir.join(src);
        if path.exists() {
            build.file(&path);
            found += 1;
        } else {
            println!("cargo:warning=Missing Lua source: {}", path.display());
        }
    }

    if found < 20 {
        println!("cargo:warning=Only found {found}/{} Lua sources / build may fail.",
                 core_sources.len());
    }

    build.compile("lua54");

    // Tell the linker to link our static library.
    println!("cargo:rustc-link-lib=static=lua54");

    // On Windows, Lua's os/io libs need these system libraries.
    if target_os == "windows" {
        // These are usually linked automatically by MSVC, but be explicit.
        // (cc crate handles most of this, but just in case.)
    }

    // On Unix, Lua needs libm and libdl.
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
    }
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=m");
    }
}

/// Compile GLSL → SPIR-V using glslc (Vulkan SDK).
///
/// Gracefully handles missing sources or missing glslc.
fn compile_shaders() {
    let shader_dir = std::path::PathBuf::from("shaders");
    let out_dir = shader_dir.join("compiled");
    std::fs::create_dir_all(&out_dir).ok();

    let shaders = [
        ("text.vert",   "text_vert.spv"),
        ("text.frag",   "text_frag.spv"),
        ("compose.comp","compose_comp.spv"),
    ];

    let glslc = find_glslc();
    if glslc.is_none() {
        println!("cargo:warning=glslc not found. Shader compilation skipped.");
        println!("cargo:warning=Install the Vulkan SDK or set VULKAN_SDK env var.");
        // Check which ones are already pre-compiled.
        for (src, dst) in &shaders {
            let dst_path = out_dir.join(dst);
            if !dst_path.exists() {
                println!("cargo:warning=Missing pre-compiled shader: {dst}");
            }
        }
        return;
    }
    let glslc = glslc.unwrap();

    for (src, dst) in &shaders {
        let src_path = shader_dir.join(src);
        let dst_path = out_dir.join(dst);

        if !src_path.exists() {
            if !dst_path.exists() {
                println!("cargo:warning=Shader {src} not found and no pre-compiled {dst}");
            }
            continue;
        }

        println!("cargo:rerun-if-changed={}", src_path.display());

        let status = std::process::Command::new(&glslc)
            .args([
                src_path.to_str().unwrap(),
                "-o", dst_path.to_str().unwrap(),
                "--target-env=vulkan1.1",
                "-O",
            ])
            .status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Compiled {src} → {dst}");
            }
            Ok(s) => println!(
                "cargo:warning=glslc failed for {src} (exit {})",
                s.code().unwrap_or(-1)
            ),
            Err(e) => println!("cargo:warning=Failed to run glslc: {e}"),
        }
    }
}

fn find_glslc() -> Option<String> {
    // 1. Check PATH.
    if let Ok(output) = std::process::Command::new("glslc")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            return Some("glslc".into());
        }
    }

    // 2. Check VULKAN_SDK env var.
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        // Standard SDK layout: $VULKAN_SDK/Bin/glslc.exe (Windows)
        //                     $VULKAN_SDK/bin/glslc     (Unix)
        for subdir in &["Bin", "bin", "Bin32"] {
            let name = if cfg!(windows) { "glslc.exe" } else { "glslc" };
            let candidate = std::path::PathBuf::from(&sdk).join(subdir).join(name);
            if candidate.exists() {
                println!("cargo:warning=Found glslc at: {}", candidate.display());
                return Some(candidate.to_string_lossy().into());
            }
        }
        println!("cargo:warning=VULKAN_SDK={sdk} but glslc not found inside it");
    }

    // 3. Windows: try common install paths.
    #[cfg(windows)]
    {
        let vulkan_paths = [
            r"C:\VulkanSDK",
        ];
        for base in &vulkan_paths {
            let base = std::path::PathBuf::from(base);
            if base.is_dir() {
                // Find the latest version directory.
                if let Ok(entries) = std::fs::read_dir(&base) {
                    let mut versions: Vec<_> = entries
                        .filter_map(|e| e.ok())
                        .filter(|e| e.path().is_dir())
                        .collect();
                    versions.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
                    for ver in versions {
                        let candidate = ver.path().join("Bin").join("glslc.exe");
                        if candidate.exists() {
                            println!("cargo:warning=Found glslc at: {}", candidate.display());
                            return Some(candidate.to_string_lossy().into());
                        }
                    }
                }
            }
        }
    }

    None
}