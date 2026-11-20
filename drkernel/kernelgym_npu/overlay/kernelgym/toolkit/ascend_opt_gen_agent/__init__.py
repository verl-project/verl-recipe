# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""AscendOptGenAgent evaluation toolkit.

Ported from ``eval_AscendOptGenAgent_original/verify.py``. Provides
multi-shape correctness evaluation with the NPU-Benchmark MERE/MARE
precision rule, while reusing the KernelBench loading and timing helpers.

Selection happens per-request via the existing ``toolkit`` field on the
evaluation request — set ``toolkit: "ascend_opt_gen_agent"`` (or set
``DEFAULT_TOOLKIT=ascend_opt_gen_agent`` on the server) to route through
this methodology instead of the default kernelbench one.
"""

from .toolkit import AscendOptGenAgentToolkit

__all__ = ["AscendOptGenAgentToolkit"]
