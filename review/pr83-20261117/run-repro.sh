#!/usr/bin/env bash
set -euo pipefail
# First prepare the toolchain described by TileOPs/docs/ascend950/toolchain.md.
# Clone TileOPs separately and checkout 0076593499b4d5f2b8c6b003f660c40e00a9eefb; do not run from baseline repo's historical TileOPs archive.
source_root=$(realpath "${1:?absolute path of a clean TileOPs PR83 checkout}")
output_root=$(realpath -m "${2:?new absolute output directory}")
expected=0076593499b4d5f2b8c6b003f660c40e00a9eefb
actual=$(git -C "$source_root" rev-parse HEAD)
[ "$actual" = "$expected" ] || { echo "Wrong source revision" >&2; exit 2; }
git -C "$source_root" diff --exit-code HEAD --
[ ! -e "$output_root" ] || { echo "Output directory must not already exist" >&2; exit 2; }
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# Keep imports, allocations, all measurements and final synchronization inside the existing host reservation.
npu-guard run --owner pr83-native-repro --wait 600 --timeout 3600 --   env -u TORCH_DEVICE_BACKEND_AUTOLOAD OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4   python "$script_dir/paired-vendor-review.py" "$source_root" "$output_root"
