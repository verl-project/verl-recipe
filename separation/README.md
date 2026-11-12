# Shared asynchronous trainer support

Used by `fully_async_policy` and `one_step_off_policy`. Inherits the V0 trainer from verl v0.9.1; `DetachActorWorker` remains in verl.

Run the CPU entrypoint checks from the pinned verl checkout:

```bash
python -m pytest recipe/separation/tests -q
```
