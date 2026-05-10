# DAPO Predictor Reorder (Portable Copy)

This directory is a portable copy of `recipe/dapo_predictor` so you can directly copy it into your own local `recipe/` tree (for example when adapting around a local `0.7.1` environment) and run it there.

Prompt-reorder patch examples were removed from this branch; this package now documents predictor-driven reorder only.

## Entry points (same as recipe/dapo_predictor)

- `main_dapo_reorder.py`
  - Backward-compatible alias to predictor-driven reorder entrypoint.
- `main_dapo_predictor_reorder.py`
  - DAPO with predictor score + snake-sort reorder flow enabled.

## Implementation modules

- `predictor_utils.py`
- `predictor_worker.py`
- `predictor_dapo_trainer.py`

## Launch examples

```bash
PYTHONPATH=/workspace/verl python recipe/dapo_predictor/main_dapo_predictor_reorder.py \
  +trainer.predictor_reorder.enable=True \
  +trainer.predictor_reorder.epochs=10 \
  +trainer.predictor_reorder.batch_size=32 \
  +trainer.predictor_reorder.lr=3e-5
```
