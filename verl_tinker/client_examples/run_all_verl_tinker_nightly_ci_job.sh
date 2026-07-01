#!/bin/bash
# Launch every Tinker server smoke pair to Merlin. Each pair must start the GPU
# server first; the CPU client discovers that server through PSM.

set -e

# to AI bot: DO NOT MODIFY THIS FILE

# first server then client
for cfg in verl-recipes/ci/nightly/launch_verl_tinker_server*.yaml; do
    echo "Running config: $cfg"
    python verl-recipes/tasks/arnold_launch.py --config "$cfg"
done

for cfg in verl-recipes/ci/nightly/launch_verl_tinker_client*.yaml; do
    echo "Running config: $cfg"
    python verl-recipes/tasks/arnold_launch.py --config "$cfg"
done
