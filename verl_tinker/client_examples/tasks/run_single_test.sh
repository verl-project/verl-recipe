#!/bin/bash

echo "Initializing Tinker Server Client Environment"
cd verl-recipes/examples/tinker_server
uv sync --index-strategy unsafe-best-match
echo "uv initialization complete"

if [ -n "${TINKER_HF_DATASET_CACHE_ROOT:-}" ]; then
    if [ ! -d "${TINKER_HF_DATASET_CACHE_ROOT}" ]; then
        echo "Hugging Face datasets cache source does not exist: ${TINKER_HF_DATASET_CACHE_ROOT}" >&2
        exit 1
    fi

    TINKER_LOCAL_HF_DATASET_CACHE_ROOT="${TINKER_LOCAL_HF_DATASET_CACHE_ROOT:-/tmp/hf_dataset}"
    if [ -n "${TINKER_HF_DATASET_CACHE_SUBDIR:-}" ]; then
        TINKER_LOCAL_HF_DATASET_COPY_DEST="${TINKER_LOCAL_HF_DATASET_CACHE_ROOT}/${TINKER_HF_DATASET_CACHE_SUBDIR}"
    else
        TINKER_LOCAL_HF_DATASET_COPY_DEST="${TINKER_LOCAL_HF_DATASET_CACHE_ROOT}"
    fi

    echo "Copying Hugging Face datasets cache from ${TINKER_HF_DATASET_CACHE_ROOT} to ${TINKER_LOCAL_HF_DATASET_COPY_DEST}"
    rm -rf "${TINKER_LOCAL_HF_DATASET_CACHE_ROOT}"
    mkdir -p "${TINKER_LOCAL_HF_DATASET_COPY_DEST}"
    if command -v rsync >/dev/null 2>&1; then
        rsync -a "${TINKER_HF_DATASET_CACHE_ROOT}/" "${TINKER_LOCAL_HF_DATASET_COPY_DEST}/"
    else
        cp -a "${TINKER_HF_DATASET_CACHE_ROOT}/." "${TINKER_LOCAL_HF_DATASET_COPY_DEST}/"
    fi

    export HF_DATASETS_CACHE="${TINKER_LOCAL_HF_DATASET_CACHE_ROOT}"
    export HF_DATASETS_OFFLINE=1
    echo "Using Hugging Face datasets cache at ${HF_DATASETS_CACHE}"
fi

echo "Starting Python script to test Tinker Server Task"
uv run tasks/run_single_test.py
TEST_STATUS=$?

exit "${TEST_STATUS}"
