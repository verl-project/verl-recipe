"""Cross-pod NIXL point-to-point bandwidth microbenchmark.

Target (pod B):    python nixl_bench.py target /workspace/bench_meta.bin
Initiator (pod A): python nixl_bench.py init /workspace/bench_meta.bin
Metadata file moves A<-B via kubectl cp; target stays alive while the
initiator times READ transfers of its 1 GiB CUDA buffer.
"""

import os
import sys
import time

import torch

from nixl._api import nixl_agent, nixl_agent_config

SIZE = 1 << 30  # 1 GiB
ITERS = 5


def _make_agent(name: str) -> nixl_agent:
    backends = os.environ.get("NIXL_BENCH_BACKENDS", "").strip()
    if backends:
        cfg = nixl_agent_config(backends=backends.split(","))
        print(f"[{name}] backends={backends}", flush=True)
        return nixl_agent(name, cfg)
    return nixl_agent(name)


def target(meta_path: str):
    agent = _make_agent("bench_target")
    buf = torch.ones(SIZE, dtype=torch.uint8, device="cuda:0")
    agent.register_memory(buf)
    descs = agent.get_xfer_descs(buf)
    blob_meta = agent.get_agent_metadata()
    blob_descs = agent.get_serialized_descs(descs)
    with open(meta_path, "wb") as f:
        f.write(len(blob_meta).to_bytes(8, "little"))
        f.write(blob_meta)
        f.write(blob_descs)
    print(f"[target] metadata written to {meta_path}; serving. Ctrl-C to stop.", flush=True)
    while True:
        # surface inbound notifications so the initiator's completion is visible
        n = agent.get_new_notifs()
        if n:
            print(f"[target] notifs: {list(n.keys())}", flush=True)
        time.sleep(0.5)


def initiator(meta_path: str):
    agent = _make_agent("bench_init")
    with open(meta_path, "rb") as f:
        mlen = int.from_bytes(f.read(8), "little")
        blob_meta = f.read(mlen)
        blob_descs = f.read()
    remote = agent.add_remote_agent(blob_meta)
    if isinstance(remote, bytes):
        remote = remote.decode()
    rdescs = agent.deserialize_descs(blob_descs)

    buf = torch.zeros(SIZE, dtype=torch.uint8, device="cuda:0")
    agent.register_memory(buf)
    ldescs = agent.get_xfer_descs(buf)

    # warmup + timed iterations
    for i in range(ITERS + 1):
        t0 = time.perf_counter()
        h = agent.initialize_xfer("READ", ldescs, rdescs, remote, b"bench")
        state = agent.transfer(h)
        while state not in ("DONE",):
            state = agent.check_xfer_state(h)
            if state == "ERR":
                print("[init] transfer ERR", flush=True)
                sys.exit(1)
        dt = time.perf_counter() - t0
        agent.release_xfer_handle(h)
        tag = "warmup" if i == 0 else f"iter{i}"
        print(f"[init] {tag}: {dt:.3f}s  {SIZE / dt / 1e9:.2f} GB/s", flush=True)
    ok = bool(buf[:1024].eq(1).all().item()) and bool(buf[-1024:].eq(1).all().item())
    print(f"[init] data check: {'OK' if ok else 'MISMATCH'}", flush=True)


if __name__ == "__main__":
    mode, path = sys.argv[1], sys.argv[2]
    os.environ.setdefault("NIXL_LOG_LEVEL", "INFO")
    (target if mode == "target" else initiator)(path)
