# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

# ruff: noqa: E501  -- long line is inside an embedded Triton kernel-code fixture string
import asyncio

import httpx


async def evaluate_kernelbench():
    reference_code = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(32, 512, device='npu')]
"""

    kernel_code = """
import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)

        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n_rows, n_cols = x.shape
        output = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        softmax_kernel[(n_rows,)](output, x, x.stride(0), output.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE)
        return output

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(32, 512, device='npu')]
"""

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8002/evaluate",
            json={
                "task_id": "softmax-kernel-003",
                "reference_code": reference_code,
                "kernel_code": kernel_code,
                "entry_point": "Model",
                "backend": "triton",
                "num_correct_trials": 5,
                "num_perf_trials": 100,
                "enable_triton_detection": True,
            },
        )
        return response.json()


result = asyncio.run(evaluate_kernelbench())
print(f"result={result}")
print(f"Compiled: {result['compiled']}")
print(f"Correctness: {result['correctness']}")
print(f"Speedup: {result['speedup']:.2f}x")
print(f"Reference Runtime: {result['reference_runtime']:.4f} ms")
print(f"Kernel Runtime: {result['kernel_runtime']:.4f} ms")
