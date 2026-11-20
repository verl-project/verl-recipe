# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

import asyncio

import httpx


async def evaluate_kernel_simple():
    timeout = httpx.Timeout(None)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "http://localhost:8002/workflow/submit",
            json={
                "workflow": "kernel_simple",
                "task_id": "my-kernel-task-011",
                "payload": {
                    "task_id": "my-kernel-task-011",
                    "kernel_code": """
import torch
import torch_npu
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

def get_init_inputs():
    return []

def get_inputs():
    x = torch.randn(1024, device='npu')
    y = torch.randn(1024, device='npu')
    return [x, y]

def get_cases():
    x = torch.randn(1024, device='npu')
    y = torch.randn(1024, device='npu')
    expected = x + y
    return [{"inputs": [x, y], "outputs": expected}]
""",
                    "entry_point": "ModelNew",
                    "backend": "triton",
                    "device": "npu:0",
                    "run_correctness": True,
                    "run_performance": True,
                    "num_perf_trials": 100,
                },
            },
        )
        return response.json()


result = asyncio.run(evaluate_kernel_simple())
print(f"result={result}")
print(f"Compiled: {result['result']['compiled']}")
print(f"Correctness: {result['result']['correctness']}")
print(f"Runtime: {result['result']['kernel_runtime']:.4f} ms")
