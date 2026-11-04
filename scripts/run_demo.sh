#!/usr/bin/env bash
# Run every real panorama bundled with GeoPano.
set -euo pipefail
GEOPANO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "${GEOPANO_ROOT}/files_inference.py" \
  --input "${GEOPANO_ROOT}/examples" \
  --output "${GEOPANO_ROOT}/outputs/real_samples" \
  "$@"
