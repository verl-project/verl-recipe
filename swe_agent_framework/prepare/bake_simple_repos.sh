#!/usr/bin/env bash
# Bake simple-test repos into the swerex Docker image.
#
# Reads simple_cases_train.json and simple_cases_val.json, creates a git repo
# for each case inside the container at /<repo_name>/, then commits the image.
#
# Usage:
#   bash bake_simple_repos.sh [IMAGE]
#
# Default image: swebenchdocker.tencentcloudcr.com/swerex/swerex-python:3.11-preinstalled

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${1:-swebenchdocker.tencentcloudcr.com/swerex/swerex-python:3.11-preinstalled}"
CONTAINER_NAME="bake_simple_repos_$$"

echo "=== Baking simple-test repos into: $IMAGE ==="

# Generate the shell commands from JSON using Python
SETUP_SCRIPT=$(python3 -c "
import json, shlex, os

script_dir = '${SCRIPT_DIR}'

commands = []
for split, fname in [('train', 'simple_cases_train.json'), ('val', 'simple_cases_val.json')]:
    path = os.path.join(script_dir, fname)
    with open(path) as f:
        cases = json.load(f)
    for idx, case in enumerate(cases):
        repo_name = f'{split}_{idx}'
        repo_dir = f'/{repo_name}'
        commands.append(f'rm -rf {repo_dir}')
        commands.append(f'mkdir -p {repo_dir}')
        for file_path, content in case.get('repo_content', {}).items():
            dir_part = os.path.dirname(file_path)
            if dir_part:
                commands.append(f'mkdir -p {repo_dir}/{dir_part}')
            escaped = shlex.quote(content if content else f'# {file_path}')
            commands.append(f'printf %s {escaped} > {repo_dir}/{file_path}')
        commands.append(f'cd {repo_dir} && git init && git add -A && git commit --allow-empty -m \"Initial commit\" >/dev/null 2>&1')

print(' && '.join(commands))
")

# Always re-bake (rm -rf inside the script handles existing repos)
echo "[*] Force re-baking all repos..."

echo "[*] Creating container and baking repos..."
docker run --name "$CONTAINER_NAME" "$IMAGE" \
    /bin/bash -c "git config --global user.email 'verl@swe-agent.local' && git config --global user.name 'VERL' && $SETUP_SCRIPT"

echo "[*] Committing image..."
docker commit "$CONTAINER_NAME" "$IMAGE" > /dev/null
docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true

# Verify
REPO_COUNT=$(docker run --rm "$IMAGE" /bin/bash -c 'ls -d /train_*/.git /val_*/.git 2>/dev/null | wc -l')
echo "=== Done: $REPO_COUNT repos baked into $IMAGE ==="
