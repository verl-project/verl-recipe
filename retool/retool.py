# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import asyncio
import logging
import re
import sys

import datasets
from recipe.retool.retool_dataset_utils import map_fn, map_fn2

from verl.tools.function_tool import function_tool
from verl.utils.dataset import RLHFDataset
from verl.utils.reward_score import math_dapo

logger = logging.getLogger(__name__)

_CODE_PATTERN = re.compile(r"```python(.*?)```", re.DOTALL)
_DEFAULT_TIMEOUT_S = 30.0


@function_tool
async def code_interpreter(code: str) -> str:
    """A tool for executing code.

    Args:
        code: The code to execute.
    """
    if not isinstance(code, str):
        code = str(code)

    matches = _CODE_PATTERN.findall(code)
    if matches:
        code = matches[0].strip()

    # NOTE: some script may not explicitly print result, we need to add a print statement to the end of the script
    lines = code.split("\n")
    for i, line in reversed(list(enumerate(lines))):
        if line == "":
            continue
        if not lines[i].startswith("print"):
            lines[i] = f"print({line})"
        break
    code = "\n".join(lines)

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-c",
        code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_DEFAULT_TIMEOUT_S)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return "TimeoutError: Code execution timed out."

    return (stdout + stderr).decode("utf-8", errors="replace")


class CustomRLHFDataset(RLHFDataset):
    """Custom dataset class to process Maxwell-Jia/AIME_2024, yentinglin/aime_2025 datasets."""

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset(parquet_file)["train"]
            data_source = "/".join(parquet_file.split("/")[-2:])
            if data_source in ["Maxwell-Jia/AIME_2024", "yentinglin/aime_2025"]:
                dataframe = dataframe.map(
                    map_fn, fn_kwargs={"data_source": data_source}, remove_columns=dataframe.column_names
                )
            else:
                dataframe = dataframe.map(map_fn2, num_proc=16)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    # use \\boxed{...} answer
    result = math_dapo.compute_score(solution_str, ground_truth, strict_box_verify=True)

    # encourage model to call tools
    num_turns = extra_info["num_turns"]
    if result["score"] < 0:
        tool_call_reward = (num_turns - 2) / 2 * 0.1
        result["score"] = min(-0.6, result["score"] + tool_call_reward)

    if result["pred"] is None:
        result["pred"] = ""

    return result
