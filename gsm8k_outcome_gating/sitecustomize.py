import os as _os

if _os.environ.get("GATE_MODE"):
    try:
        import gate_hook as _gh

        _gh.install()
    except Exception as _e:
        print("[gate] install failed:", _e, flush=True)
        raise
