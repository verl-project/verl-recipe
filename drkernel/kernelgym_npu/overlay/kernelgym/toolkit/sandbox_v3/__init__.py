# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""sandbox_v3 evaluation toolkit.

Vendored from the ``Triton-Training-kernelgym-sandbox`` KernelBench toolkit and
hard-wired to the NPU-kernel evaluation path (``NPUKERNEL_MODE=on`` /
``ORIGIN_MODE=off`` in the original): multi-shape correctness via
``get_input_groups()``, allclose-style precision, AST-based Triton-
implementation validation (``fillKernelExecResult``), and operator-level NPU
profiling (``measure_single``).

Selection happens per-request via the existing ``toolkit`` field on the
evaluation request — set ``toolkit: "sandbox_v3"`` (or set
``DEFAULT_TOOLKIT=sandbox_v3`` on the server) to route through this toolkit.

Keep this module import-light: the toolkit class is imported lazily by the
toolkit registry, so importing heavy modules here can create circular imports.
"""

__all__ = []
