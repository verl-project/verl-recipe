#!/usr/bin/env bash
# ==============================================================================
# Wire the gate into a pinned-verl installation. Two small, reversible steps:
#
#  1. Drop zzz_gate_path.pth into the venv's site-packages so every process —
#     including Ray workers — imports this directory's sitecustomize.py, which
#     installs gate_hook when GATE_MODE is set.
#     Why not monkey-patch from the driver? compute_advantage runs inside the
#     TaskRunner Ray actor (a separate process); a driver-side patch never
#     reaches it. The .pth + sitecustomize route is what actually lands there.
#
#  2. Insert a 6-line env-gated branch at the top of verl's built-in gsm8k
#     reward (backed up to gsm8k.py.bak). Why not data.custom_reward_function?
#     At the pinned commit the async agent-loop reward path (RewardLoopWorker)
#     bypasses it, so the built-in must carry the branch. The branch is inert
#     unless GATE_SHAPED is set.
#
# Usage:   ./install_gate.sh /path/to/venv/bin/python
# Undo:    restore gsm8k.py.bak and delete zzz_gate_path.pth from site-packages.
# ==============================================================================
set -euo pipefail
VENV_PY=${1:?usage: install_gate.sh /path/to/venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)

SITE=$("$VENV_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "import sys; sys.path.insert(0, \"$HERE\")" > "$SITE/zzz_gate_path.pth"
echo "[install_gate] wrote $SITE/zzz_gate_path.pth"

GSM=$("$VENV_PY" - <<'PY'
import verl.utils.reward_score.gsm8k as m
print(m.__file__)
PY
)
if grep -q "GATE_SHAPED" "$GSM"; then
  echo "[install_gate] reward branch already present in $GSM"
else
  cp "$GSM" "$GSM.bak"
  "$VENV_PY" - "$GSM" "$HERE" <<'PY'
import re
import sys

path, here = sys.argv[1], sys.argv[2]
src = open(path).read()
m = re.search(r"^def compute_score\(.*\):\n", src, flags=re.M)
if not m:
    raise SystemExit(f"[install_gate] could not find 'def compute_score(...):' in {path}; patch by hand")
branch = (
    f"    import os as _os\n"
    f"    if _os.environ.get(\"GATE_SHAPED\"):\n"
    f"        import sys as _sys\n"
    f"        if \"{here}\" not in _sys.path: _sys.path.insert(0, \"{here}\")\n"
    f"        import gate_shaped_reward as _gsr\n"
    f"        return _gsr.compute_score(\"openai/gsm8k\", solution_str, ground_truth)\n\n"
)
src = src[: m.end()] + branch + src[m.end() :]
open(path, "w").write(src)
print(f"[install_gate] patched {path} (backup: {path}.bak)")
PY
fi
echo "[install_gate] done. The gate is inert unless GATE_SHAPED / GATE_MODE are set."
