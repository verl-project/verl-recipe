"""
Shaped GSM8K reward for the outcome-gating "correct test".

    reward = outcome(0/1)  -  LAMBDA * norm_length

The length term is the phantom: among FAILURES it rewards the *shortest* wrong answer (give up
faster) — the documented Goodhart trap (the search-agent's 24→7-char answer-length collapse). So
an all-fail group has std>0 (different lengths → different reward), which DAPO's std filter keeps
and GRPO turns into a phantom gradient; the binary-outcome gate drops it.

Exposes `outcome_binary` (the UNSHAPED correctness) via the score dict so the gate groups on the
real outcome, not the shaped scalar. veRL's reward manager returns the scalar as the token-level
score; the extra key rides along in extra_info handling where supported, else the gate falls back
to thresholding — but we ALSO stash it so the run log can be audited.

compute_score(data_source, solution_str, ground_truth, extra_info=None) -> dict|float
"""

import os
import re

LAMBDA = float(os.environ.get("GATE_LAMBDA", "0.30"))  # phantom strength; env-swept for dose-response
PHANTOM = os.environ.get("GATE_PHANTOM", "short")  # short: reward brevity in failures / long: reward verbosity
LEN_CAP = 512  # chars; responses at/above this get full length penalty


def _extract(text):
    """Flexible GSM8K answer extraction: #### , \\boxed{}, 'answer is', else last number."""
    if text is None:
        return None
    for pat in (
        r"####\s*([+-]?[\d,]+(?:\.\d+)?)",
        r"\\boxed\{\s*([+-]?[\d,]+(?:\.\d+)?)\s*\}",
        r"(?:answer|result)\s*(?:is|:|=)\s*\$?\s*([+-]?[\d,]+(?:\.\d+)?)",
    ):
        m = re.findall(pat, text, flags=re.IGNORECASE)
        if m:
            return m[-1].replace(",", "").rstrip(".")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def _correct(solution_str, ground_truth):
    pred = _extract(solution_str)
    if pred is None:
        return 0
    gold = str(ground_truth).replace(",", "").strip()
    gold = _extract(gold) or gold
    try:
        return int(abs(float(pred) - float(gold)) < 1e-4)
    except ValueError:
        return int(pred == gold)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    outcome = _correct(solution_str, ground_truth)
    norm_len = min(len(solution_str or "") / LEN_CAP, 1.0)
    pen = norm_len if PHANTOM == "short" else (1.0 - norm_len)
    shaped = float(outcome) - LAMBDA * pen
    # Return a dict so the outcome rides along; veRL uses "score" as the scalar reward.
    return {"score": shaped, "outcome_binary": outcome, "acc": float(outcome)}
