# AGENTS.md

Guidance for OpenCode sessions working in this repo.

## What this repo is

Shared CMake/Python build, packaging, and install framework for the CANN ecosystem. It is **not a standalone buildable project** — it is consumed by sibling CANN component repos (runtime, metadef, ge, etc.) checked out alongside it under a common parent (`CANN_TOP_DIR` = parent of this repo). There is no root `CMakeLists.txt`; the build entry point is `superbuild/CMakeLists.txt`, driven by `build.sh`.

## Build

```bash
# --pkgs is REQUIRED. Comma-separated package names (e.g. runtime,asc-devkit).
sh build.sh --pkgs=<PACKAGES> [-j<N>] [-v] [--pkg-type=run|rpm|deb|deb,rpm|all] [--build-type=Release|Debug] [--build_host_only] [--asan]
```

- Build dir: `build/`. Output (packaged artifacts): `build_out/`.
- `build.sh` runs `cmake -S superbuild -B build` then `cmake --build build --target package`.
- A full build requires the sibling CANN repos present under `CANN_TOP_DIR` and a CANN toolkit install. Without these the superbuild cannot resolve dependencies — do not attempt to "fix" this by making the repo standalone.
- `ASCEND_CANN_PACKAGE_PATH` (toolkit install) resolution order in `build.sh`: `-p/--cann_path` → `ASCEND_HOME_PATH` → `ASCEND_OPP_PATH` → default dirs (`~/Ascend/cann` non-root, `/usr/local/Ascend/cann` root).
- `--build_host_only` skips the device cross-compile step by setting `ENABLE_BUILD_DEVICE=FALSE`; when `build.sh` is not used, `ENABLE_BUILD_DEVICE` defaults to `TRUE` (see host/device split below).

## Tests

```bash
python3 -m pytest scripts/                        # all tests (package, version, sign, build_analysis)
python3 -m pytest scripts/package/tests/          # packaging tests (from repo root)
python3 -m pytest scripts/version/tests/          # version info generation tests
python3 -m pytest scripts/build_analysis/tests/   # IWYU log parser tests
python3 -m pytest scripts/sign/tests/             # signing script tests
python3 -m pytest scripts/package/tests/test_package.py::TestClass -v   # single test
```

- No `pytest.ini`/`pyproject.toml`. Each test dir has its own `conftest.py` that prepends its parent to `sys.path` so modules import without install.
- `classify_rule.yaml` sets `llt.ut_check: true` — keep tests green when changing `scripts/package/**`.

## Pre-commit

```bash
pre-commit install          # one-time setup, activates git hook
pre-commit run --all-files  # manual full-repo check
```

- Config in `.pre-commit-config.yaml`. Before committing, run `pre-commit run --files <changed files>` to verify hooks pass.
- Hooks: trailing-whitespace, end-of-file-fixer, check-yaml, check-added-large-files, check-merge-conflict, detect-private-key, check-json (pre-commit-hooks); ruff-check + ruff-format (Python); pylint R1710/R0801/W0621 (rules ruff cannot cover, config in `.pylintrc`); codespell; OAT compliance (`scripts/oat_check.sh`).
- Some hooks auto-fix (trailing-whitespace, end-of-file-fixer, ruff-check `--fix`, ruff-format); re-stage after they run.
- Existing code may have pre-existing violations — only files you touch are checked at commit time.

## Architecture notes that are not obvious from filenames

- **Two integration modes, gated by `TOPLEVEL_PROJECT`** (`function/prepare.cmake:13-19`): ON only when `prepare.cmake` is included *before* `project()` (single-repo standalone build, per the README integration pattern). In the superbuild, `project(cann_superbuild)` runs first, so `TOPLEVEL_PROJECT` is OFF; `build.sh` always passes `ENABLE_UNIFIED_BUILD=TRUE`, which re-enables functions guarded by `TOPLEVEL_PROJECT OR ENABLE_UNIFIED_BUILD` (`set_cann_cpack_config`, `add_cann_third_party`, `find_cann_package`). Functions guarded by `TOPLEVEL_PROJECT` only (`add_cann_device_project`, `check_cann_pkg_build_deps`) stay disabled in superbuild mode — the superbuild handles device builds via its own `ExternalProject_Add` in `superbuild/CMakeLists.txt`. Always check each function's guard condition when editing `function/prepare.cmake`.
- **`CANN_TOP_DIR`** is resolved as the parent of this repo (`function/function.cmake:31`). Package dependency resolution reads `version.cmake` files from `${CANN_TOP_DIR}/${pkg_dir}` — `superbuild/config.cmake` maps both package→source dir (`CANN_PACKAGE_DIR_*`) and package→components (`CANN_PACKAGE_COMPONENTS_*`); update both when adding a package.
- **Device cross-compile** uses `toolchain/aarch64-hcc-toolchain.cmake` (aarch64 target via hcc; compiler is `${TOOLCHAIN_DIR}/bin/aarch64-target-linux-gnu-gcc`, where `TOOLCHAIN_DIR` defaults to `${ASCEND_INSTALL_PATH}/toolkit/toolchain/hcc`). Host builds compile in-process via `add_subdirectory` of each sibling repo under `CANN_TOP_DIR`; only device builds run as a separate `ExternalProject_Add` step. `PRODUCT_SIDE` (`host`/`device`) controls which source dirs are added — host adds `<pkg>`, device adds `<pkg>/cmake/device` (the `cmake/device` subdir arg passed to `get_package_dirs` at `superbuild/device/CMakeLists.txt:17`). Device cross-compile is auto-skipped when no selected package has a `cmake/device` subdir; `calc_device_packages` also excludes `ge-executor`/`ge-compiler` (pure host-side packages sharing the `ge` repo with `dflow-executor`) so they never trigger device build. `--build_host_only` forces it off via `ENABLE_BUILD_DEVICE=FALSE`; when `build.sh` is not used, `ENABLE_BUILD_DEVICE` defaults to `TRUE` (`superbuild/CMakeLists.txt:43-47`).
- **`__FILE__` macro rewriting:** Both `prepare.cmake` (non-Ninja) and the toolchain file override `CMAKE_C/CXX_COMPILE_OBJECT` to emit only the basename in `__FILE__`, plus `-Wno-builtin-macro-redefined`. Preserve this when touching compile rules.
- **ABI:** `CANN_CXX11_ABI` is computed once in `init_cann_project()` (`function/prepare.cmake`) and consumed by `intf_pub` + all third-party scripts. Host defaults to `_GLIBCXX_USE_CXX11_ABI=0` (old ABI); device defaults to `1`. `USE_CXX11_ABI` overrides either. See `intf_pub/intf_pub_linux.cmake:57`.
- **ccache** is on by default (`ENABLE_CCACHE`).
- **Dependency pairs are stride-2 lists:** `set_cann_build_dependencies`/`set_cann_run_dependencies` append `pkg;version` pairs to `CANN_VERSION_*_BUILD_DEPS`/`_RUN_DEPS`. Consumers (`get_build_pkg_deps` in `function/function.cmake`, `scripts/version/generate_version_info.py`) walk the list 2 at a time, so changing the call format requires updating both (flagged in code comments).
- **Never set `EXCLUDE_FROM_ALL` on `add_subdirectory`** — CMake then skips that dir's `install()` rules, silently dropping components from packages. `set_cann_all_targets` (`function/function.cmake:182`) instead marks individual *targets* `EXCLUDE_FROM_ALL` and aggregates wanted ones under `cann_all_targets`; the `add_subdirectory` calls in `superbuild/CMakeLists.txt:67` deliberately omit it for the same reason.

## Public API consumed by CANN repos

Functions in `function/prepare.cmake` are the framework's public API: `init_cann_project`, `set_cann_package`, `set_cann_build_dependencies`, `set_cann_run_dependencies`, `check_cann_build_tool`, `set_cann_cpack_config`, `add_cann_third_party`, `find_cann_package`, `generate_cann_stub_library`, `cann_pack_targets_and_files`, `gen_cann_version_header`, `add_cann_sign_file`. Consumer repos call these from their own `CMakeLists.txt`; changing signatures or behavior is a cross-repo breaking change.

## Directory map

| Path | Purpose |
|------|---------|
| `function/` | Core framework: `prepare.cmake` (public API), `function.cmake` (superbuild init + dep resolution) |
| `superbuild/` | Superbuild entrypoints (`CMakeLists.txt` host, `device/CMakeLists.txt` device, `config.cmake` package→dir map) |
| `modules/` | `Find<dep>.cmake` modules; added to `CMAKE_MODULE_PATH` via `PREPEND_MODULE_PATH` |
| `third_party/` | `<name>.cmake` scripts for building external deps (abseil, boost, grpc, protobuf, ...), included via `add_cann_third_party(name)` |
| `intf_pub/` | `intf_pub_linux.cmake` — shared compile/link flags (security hardening, sanitizers, ABI) applied via `add_cann_target_options` |
| `toolchain/` | `aarch64-hcc-toolchain.cmake` for device cross-compile |
| `docs/` | Detailed architecture/integration docs (`architecture.md`, `framework/`, `superbuild/`, `install/`) |
| `scripts/package/` | Python packaging logic + pytest suite; invoked by CMake at `package` time |
| `scripts/version/` | `check_build_dependencies.py`, `generate_version_info.py` — invoked by CMake functions; pytest suite in `tests/` |
| `scripts/sign/` | Code signing scripts invoked by `add_cann_sign_file`; pytest suite in `tests/` |
| `scripts/signtool/` | Python image pack/extract/ESBC-header tools used by the sign flow |
| `scripts/build_analysis/` | IWYU log parser (`iwyu_log_parser.py`) + pytest suite in `tests/` |
| `scripts/install/` | Shell install scripts shipped in the packages; permission behavior must follow `docs/install/permissions.md` (install permission spec) |

## Conventions

- **License header required** on every source file (CANN-2.0), enforced by `OAT.xml`. Copy the header block from any existing file when creating new `.cmake`/`.py`/`.sh` files.
- **C++ standard is C++17** (`CMAKE_CXX_STANDARD 17`, no extensions), set in `init_cann_project`.
- Comments throughout the framework are in Chinese — match this when editing existing files.
- **Docs must stay in sync with code changes:** when modifying `build.sh` parameters/options, `superbuild/CMakeLists.txt` behavior, `config.cmake` package mappings, or any public API in `function/prepare.cmake`, update the corresponding docs (`docs/superbuild/getting-started.md`, `docs/superbuild/internals.md`, `docs/framework/public-api.md`, `AGENTS.md` Build/Architecture sections) in the same change. Do not leave docs describing stale behavior. When changing install/permission logic in `scripts/install/**` or `scripts/package/**`, update `docs/install/permissions.md` in the same change.
- Consumer integration pattern (see `README.md`): `fetch_cann_cmake.cmake` → `include(function/prepare.cmake)` → `init_cann_project()` after `project()`.
