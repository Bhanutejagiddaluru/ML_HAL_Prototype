
#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="${HERE}/.."
EXT_DIR="${ROOT}/src/hal/cpp"

mkdir -p "${EXT_DIR}/build"
cd "${EXT_DIR}/build"
cmake -DPYBIND11_FINDPYTHON=ON ..
cmake --build . --config Release

# Copy built module next to Python so Python can import 'hal_ext'
OUT=$(python - <<'PY'
import sys, site, pathlib
print(site.getusersitepackages())
PY
)
mkdir -p "$OUT"
cp ./hal_ext*.so "$OUT" 2>/dev/null || true
cp ./Release/hal_ext*.pyd "$OUT" 2>/dev/null || true
echo "Installed hal_ext to: $OUT"
