#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Set up the KernelGYM reward server with Ascend NPU support.
#
# Clones upstream hkust-nlp/KernelGYM at the pinned commit, applies the NPU
# adaptation patch (patches/), and copies in the files new to this recipe
# (overlay/: NPU toolkits, tests, .env template).
#
# Usage:
#   bash setup_kernelgym.sh [target_dir]      # default: ./KernelGYM

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

KERNELGYM_REPO="${KERNELGYM_REPO:-https://github.com/hkust-nlp/KernelGYM.git}"
KERNELGYM_COMMIT="${KERNELGYM_COMMIT:-3a84417f8c0efaadb215ef638b37d12e71ed20f3}"
TARGET_DIR="${1:-${SCRIPT_DIR}/KernelGYM}"

if [ -e "${TARGET_DIR}" ]; then
    echo "error: ${TARGET_DIR} already exists; remove it or pass a different target dir." >&2
    exit 1
fi

git clone "${KERNELGYM_REPO}" "${TARGET_DIR}"
git -C "${TARGET_DIR}" checkout --quiet "${KERNELGYM_COMMIT}"

# Upstream has stale __pycache__ dirs committed; drop them.
find "${TARGET_DIR}" -type d -name __pycache__ -prune -exec rm -rf {} +

git -C "${TARGET_DIR}" apply --whitespace=nowarn "${SCRIPT_DIR}/patches/kernelgym-npu-support.patch"

cp -R "${SCRIPT_DIR}/overlay/." "${TARGET_DIR}/"

echo
echo "KernelGYM (NPU) set up at: ${TARGET_DIR}"
echo "Next steps:"
echo "  1. cd ${TARGET_DIR} && bash setup.sh     # install python deps + redis"
echo "  2. cp .env.example .env                  # then edit API_HOST / GPU_DEVICES"
echo "  3. bash start_all_with_monitor.sh        # start the server"
