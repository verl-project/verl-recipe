#!/usr/bin/env bash
# Install only the advantage-hook import path in a dedicated virtual environment.
# Reward scoring uses reward.custom_reward_function.path; no verl source is edited.
# Undo: remove zzz_gate_path.pth from that environment's site-packages.
set -euo pipefail
VENV_PY=${1:?usage: install_gate.sh /path/to/venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
"$VENV_PY" - "$HERE" <<'PYTHON'
from pathlib import Path
import sys
import sysconfig

import verl.utils.reward_score.gsm8k as gsm8k

source = Path(gsm8k.__file__)
if "GATE_SHAPED" in source.read_text():
    raise SystemExit(
        f"Legacy reward patch found in {source}. Review it against {source}.bak "
        "and restore the original before reinstalling; this installer never overwrites core source."
    )
site = Path(sysconfig.get_paths()["purelib"])
entry = site / "zzz_gate_path.pth"
entry.write_text(f"import sys; sys.path.insert(0, {sys.argv[1]!r})\n")
print(f"[install_gate] wrote {entry}; verl source unchanged")
PYTHON
