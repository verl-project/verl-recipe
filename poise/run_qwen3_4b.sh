#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec python -m recipe.poise.main --config-name qwen3_4b "$@"
