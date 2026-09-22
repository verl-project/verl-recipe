# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

# DR.Kernel-style multi-turn agent loop for verl.
#
# Implements DR.Kernel's prompt-template multi-turn behavior on top of
# verl's `AgentLoopBase` infrastructure. After each assistant turn, the
# kernel code is extracted, evaluated against KernelGym, and the result
# is folded back as a templated user message that drives the next turn.
#
# See `kernel_agent_loop.KernelAgentLoop` for the entry point.
