# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Thin wrapper around ``python -m dynamo.vllm`` that adds a verl-private
control sidecar.

verl × dynamo's weight-update path needs to invoke
``engine.collective_rpc("update_weights_from_ipc", ...)`` from outside the
``dynamo.vllm`` subprocess (the trigger comes from the trainer side; the
actor in our recipe doesn't hold an AsyncLLM). Bare ``dynamo.vllm`` exposes
no such hook. This wrapper:

  1. Patches ``dynamo.vllm.main.setup_vllm_engine`` to capture the AsyncLLM
     instance into a module-level holder.
  2. Spawns a ZMQ REP listener (endpoint from ``$VERL_DYNAMO_CONTROL_ZMQ``)
     that bridges:

       * ``{"kind": "collective_rpc", "method": <name>, "args": ..., "kwargs": ..., "timeout": ...}``
         → ``engine.collective_rpc(method, timeout=..., args=..., kwargs=...)``

       * ``{"kind": "engine_method", "method": <name>, "kwargs": ...}``
         → ``getattr(engine, method)(**kwargs)`` (awaits if coroutine)

  3. Calls the standard ``dynamo.vllm.main.worker()`` flow.

The actor's ``DynamoHttpServer.collective_rpc`` and
``DynamoHttpServer._engine_method_all`` are the two REQ-side users.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import sys
import traceback
from typing import Any, Optional

logger = logging.getLogger("recipe.dynamo._wrapper")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [dynamo-vllm-control] %(levelname)s: %(message)s"))
    logger.addHandler(handler)

# Module-level holder so the patched setup_vllm_engine + control listener
# can rendezvous on the same AsyncLLM instance.
_engine_holder: dict[str, Any] = {"engine": None, "vllm_config": None}


def _install_engine_capture():
    """Wrap ``dynamo.vllm.main.setup_vllm_engine`` so we get the AsyncLLM."""
    import dynamo.vllm.main as dyn_main

    original = dyn_main.setup_vllm_engine

    # Defensive injection: also force worker_extension_cls if caller didn't
    # already pass --worker-extension-cls. Most callers (DynamoHttpServer)
    # already pass it via CLI but we want this module to be robust if
    # someone runs us directly.
    default_ext = "recipe.dynamo.dynamo_worker_extension.vLLMDynamoColocateWorkerExtension"

    def patched(config, *args, **kwargs):
        engine_args = getattr(config, "engine_args", None)
        if engine_args is not None and not getattr(engine_args, "worker_extension_cls", None):
            engine_args.worker_extension_cls = default_ext
            logger.info("injected worker_extension_cls=%s", default_ext)
        result = original(config, *args, **kwargs)
        # result = (engine_client, vllm_config, default_sampling_params,
        #           prometheus_temp_dir, component_gauges)
        try:
            engine_client = result[0]
            vllm_config = result[1]
            _engine_holder["engine"] = engine_client
            _engine_holder["vllm_config"] = vllm_config
            logger.info(
                "captured AsyncLLM (model=%s)",
                getattr(getattr(vllm_config, "model_config", None), "model", "?"),
            )
        except Exception:
            logger.exception("failed to capture AsyncLLM from setup_vllm_engine result")
        return result

    dyn_main.setup_vllm_engine = patched


async def _wait_for_engine(timeout: float = 1800.0) -> Any:
    """Block until setup_vllm_engine has been called and captured the engine."""
    deadline = asyncio.get_event_loop().time() + timeout
    while _engine_holder["engine"] is None:
        if asyncio.get_event_loop().time() > deadline:
            raise TimeoutError(
                f"engine_client was not captured within {timeout}s; check that setup_vllm_engine was actually invoked"
            )
        await asyncio.sleep(0.5)
    return _engine_holder["engine"]


async def _handle_request(req: dict) -> dict:
    """Dispatch one control request to the captured engine."""
    kind = req.get("kind", "collective_rpc")
    method = req.get("method")
    args = tuple(req.get("args") or ())
    kwargs = dict(req.get("kwargs") or {})
    timeout = req.get("timeout")

    engine = _engine_holder["engine"]
    if engine is None:
        return {"ok": False, "error": "engine not yet ready"}

    try:
        if kind == "collective_rpc":
            result = await engine.collective_rpc(method=method, timeout=timeout, args=args, kwargs=kwargs)
        elif kind == "engine_method":
            fn = getattr(engine, method)
            ret = fn(**kwargs)
            if asyncio.iscoroutine(ret):
                if timeout is not None:
                    ret = await asyncio.wait_for(ret, timeout=timeout)
                else:
                    ret = await ret
            result = ret
        elif kind == "generate_direct":
            # Bypass dynamo's HTTP/frontend stack and call AsyncLLM.generate
            # directly. Used by DynamoHttpServer.generate as a fallback when
            # the dynamo /v1/chat/completions path hangs (observed in
            # ai-dynamo 1.0.2 + vllm 0.17). Trades the KV-router benefit for
            # a working request path.
            from vllm import SamplingParams
            from vllm.inputs import TextPrompt, TokensPrompt  # noqa: F401

            token_ids = list(kwargs.get("token_ids") or [])
            sp_kwargs = dict(kwargs.get("sampling_params") or {})
            request_id = kwargs.get("request_id") or "direct-no-id"
            prompt_text = kwargs.get("prompt_text")  # optional; preferred path

            # Filter sp_kwargs to keys that SamplingParams actually accepts.
            # vLLM's SamplingParams uses pydantic, so __init__ is a wrapper
            # descriptor without __code__. inspect.signature works on most
            # versions; if even that fails, drop unknown keys progressively.
            import inspect

            try:
                sp_accepts = set(inspect.signature(SamplingParams).parameters.keys())
            except (TypeError, ValueError):
                sp_accepts = None
            if sp_accepts is not None:
                sp_filtered = {k: v for k, v in sp_kwargs.items() if k in sp_accepts and v is not None}
            else:
                sp_filtered = {k: v for k, v in sp_kwargs.items() if v is not None}
            try:
                sampling_params = SamplingParams(**sp_filtered)
            except TypeError as e:
                # progressively drop unknown keys mentioned in the error
                msg = str(e)
                logger.warning("SamplingParams init failed: %s; retrying", msg)
                for bad in ("logprobs", "top_k", "repetition_penalty", "seed"):
                    sp_filtered.pop(bad, None)
                sampling_params = SamplingParams(**sp_filtered)

            # Prefer TextPrompt — TokensPrompt path triggers a "raw prompts"
            # deprecation warning in vLLM 0.17 and observed to hang inside
            # dynamo's intercepted generate. TextPrompt goes through a
            # different code path that works.
            if prompt_text:
                prompt = TextPrompt(prompt=prompt_text)
            else:
                prompt = TokensPrompt(prompt_token_ids=token_ids)
            all_token_ids: list[int] = []
            num_emitted = 0
            finish_reason: Optional[str] = None
            try:
                async for output in engine.generate(
                    prompt=prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                ):
                    if not output.outputs:
                        continue
                    res = output.outputs[0]
                    all_token_ids = list(res.token_ids)
                    num_emitted = len(all_token_ids)
                    if res.finish_reason:
                        finish_reason = res.finish_reason
            except asyncio.CancelledError:
                raise
            except Exception as e:
                return {
                    "ok": False,
                    "error": f"engine.generate raised: {type(e).__name__}: {e}",
                }
            result = {
                "token_ids": all_token_ids,
                "num_emitted": num_emitted,
                "finish_reason": finish_reason,
            }
        else:
            return {"ok": False, "error": f"unknown kind: {kind}"}
    except Exception as e:
        logger.exception("control request failed: kind=%s method=%s", kind, method)
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
    # Some results (e.g. SamplingParams, vLLM internal objects) may not be
    # picklable. Fall back to repr() if pickle fails.
    try:
        pickle.dumps(result)
        return {"ok": True, "result": result}
    except Exception:
        return {"ok": True, "result": repr(result), "result_was_repr": True}


async def _control_listener(endpoint: str):
    """ZMQ REP loop on `endpoint`. One in-flight request at a time —
    AsyncLLM.collective_rpc is itself a synchronization point, no benefit to
    pipelining."""
    import zmq
    import zmq.asyncio

    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(endpoint)
    logger.info("control listener bound to %s", endpoint)

    # Wait for the engine to be alive before serving requests.
    await _wait_for_engine()

    try:
        while True:
            try:
                raw = await sock.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("recv failed")
                continue
            try:
                req = pickle.loads(raw)
            except Exception as e:
                await sock.send(pickle.dumps({"ok": False, "error": f"bad pickle: {e}"}))
                continue
            reply = await _handle_request(req)
            try:
                await sock.send(pickle.dumps(reply))
            except Exception:
                # If the reply itself isn't sendable, downgrade to a simple ok.
                logger.exception("send failed; downgrading reply")
                try:
                    await sock.send(pickle.dumps({"ok": reply.get("ok", False), "error": "reply was not picklable"}))
                except Exception:
                    pass
    finally:
        sock.close(0)
        ctx.term()


async def _amain():
    import dynamo.vllm.main as dyn_main

    _install_engine_capture()

    control_ep = os.environ.get("VERL_DYNAMO_CONTROL_ZMQ")
    if control_ep:
        listener = asyncio.create_task(_control_listener(control_ep))
    else:
        logger.warning(
            "VERL_DYNAMO_CONTROL_ZMQ is not set; control sidecar disabled "
            "(weight update / sleep / wake will not work for this shard)"
        )
        listener = None

    try:
        await dyn_main.worker()
    finally:
        if listener is not None:
            listener.cancel()
            try:
                await listener
            except (asyncio.CancelledError, Exception):
                pass


def main():
    try:
        import uvloop  # type: ignore

        uvloop.run(_amain())
    except ImportError:
        asyncio.run(_amain())


if __name__ == "__main__":
    main()
