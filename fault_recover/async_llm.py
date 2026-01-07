# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

import numpy as np
from vllm.envs import VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
from vllm.utils import cdiv
from vllm.v1.engine.async_llm import AsyncLLM, logger
from vllm.v1.metrics.stats import IterationStats


class AsyncFaultRecoverLLM(AsyncLLM):
    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        log_stats = self.log_stats
        logger_manager = self.logger_manager

        async def output_handler(q):
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()

                    if q is not None:
                        req_info = {}
                        for output in outputs.outputs:
                            req_info[output.request_id] = {}
                            req_info[output.request_id]["new_token_ids"] = output.new_token_ids
                            req_info[output.request_id]["finished"] = output.finished
                        await q.put.remote(req_info)

                    num_outputs = len(outputs.outputs)

                    iteration_stats = IterationStats() if (log_stats and num_outputs) else None

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    if num_outputs <= VLLM_V1_OUTPUT_PROC_CHUNK_SIZE:
                        slices = (outputs.outputs,)
                    else:
                        slices = np.array_split(outputs.outputs, cdiv(num_outputs, VLLM_V1_OUTPUT_PROC_CHUNK_SIZE))

                    for i, outputs_slice in enumerate(slices):
                        # 2) Process EngineCoreOutputs.
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        if i + 1 < len(slices):
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        await engine_core.abort_requests_async(processed_outputs.reqs_to_abort)

                    # 4) Logging.
                    # TODO(rob): make into a coroutine and launch it in
                    # background thread once Prometheus overhead is non-trivial.
                    if logger_manager:
                        logger_manager.record(
                            engine_idx=outputs.engine_index,
                            scheduler_stats=outputs.scheduler_stats,
                            iteration_stats=iteration_stats,
                        )
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        from recipe.fault_recover.fault_manager import get_tokens_queue

        tokens_queue = get_tokens_queue()

        self.output_handler = asyncio.create_task(output_handler(tokens_queue))
