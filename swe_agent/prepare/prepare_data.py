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
"""
SWE Agent Dataset Generator

Supports two data sources:
1. Simple test cases (for quick validation)
2. SWE-bench Lite (for full evaluation)

Data format is VERL-compatible:
- prompt: Conversation format containing problem_statement
- reward_model: Evaluation configuration
- extra_info: Contains repo_content, expected_patch, etc.
- agent_name: "swe_agent"
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd


# SWE-Agent system prompt
SWE_AGENT_SYSTEM_PROMPT = """You are a helpful assistant that can interact with a computer to solve software engineering tasks.

When working on a task:
1. First understand the problem by reading relevant code
2. Make minimal changes to fix the issue
3. Verify your changes work correctly

Your goal is to create a patch that resolves the issue described in the problem statement."""


def create_swe_prompt(problem_statement: str, repo_info: Optional[Dict] = None) -> List[Dict[str, str]]:
    """Create SWE-Agent prompt format.
    
    Args:
        problem_statement: Problem description.
        repo_info: Optional repository information.
        
    Returns:
        Prompt in conversation format.
    """
    user_content = f"""<problem_statement>
{problem_statement}
</problem_statement>

Please analyze the problem and implement the necessary changes to resolve it.
Make minimal changes to the codebase while ensuring the issue is fully addressed."""

    if repo_info:
        repo_name = repo_info.get("repo_name", "unknown")
        base_commit = repo_info.get("base_commit", "")
        user_content = f"""Repository: {repo_name}
Base commit: {base_commit}

{user_content}"""

    return [
        {"role": "system", "content": SWE_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_simple_test_data(num_samples: int, split: str, agent_name: str = "swe_agent") -> pd.DataFrame:
    """Generate simple test data (for quick validation).
    
    Args:
        num_samples: Number of samples.
        split: Dataset split (train/test).
        agent_name: Agent name.
        
    Returns:
        Dataset in DataFrame format.
    """
    rl_dataset = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
        "agent_name": [],
    }
    
    # Simple test case: file rename
    test_cases = [
        {
            "problem_statement": "rename 1.txt to 2.txt",
            "repo_content": {"1.txt": "Hello World"},
            "expected_patch": """diff --git a/1.txt b/2.txt
similarity index 100%
rename from 1.txt
rename to 2.txt""",
        },
        {
            "problem_statement": "Create a new file called hello.py that prints 'Hello, World!'",
            "repo_content": {},
            "expected_patch": """diff --git a/hello.py b/hello.py
new file mode 100644
--- /dev/null
+++ b/hello.py
@@ -0,0 +1 @@
+print('Hello, World!')""",
        },
        {
            "problem_statement": "Fix the bug in calculator.py: the add function should return a + b, not a - b",
            "repo_content": {"calculator.py": "def add(a, b):\n    return a - b"},
            "expected_patch": """diff --git a/calculator.py b/calculator.py
--- a/calculator.py
+++ b/calculator.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b""",
        },
    ]
    
    for idx in range(num_samples):
        case = test_cases[idx % len(test_cases)]
        
        # Create prompt
        prompt = create_swe_prompt(case["problem_statement"])
        
        # Reward model config
        reward_model = {
            "style": "swe_agent",
            "ground_truth": case["expected_patch"],
        }
        
        # Extra info
        extra_info = {
            "index": idx,
            "split": split,
            "repo_content": case["repo_content"],
            "expected_patch": case["expected_patch"],
            "problem_statement": case["problem_statement"],
        }
        
        rl_dataset["prompt"].append(prompt)
        rl_dataset["data_source"].append("swe_agent_simple")
        rl_dataset["ability"].append("software_engineering")
        rl_dataset["reward_model"].append(reward_model)
        rl_dataset["extra_info"].append(extra_info)
        rl_dataset["agent_name"].append(agent_name)
    
    return pd.DataFrame(data=rl_dataset)


def load_swebench_lite(swebench_path: str, split: str, agent_name: str = "swe_agent") -> pd.DataFrame:
    """Load SWE-bench Lite dataset.
    
    Args:
        swebench_path: Path to SWE-bench data file.
        split: Dataset split.
        agent_name: Agent name.
        
    Returns:
        Dataset in DataFrame format.
    """
    rl_dataset = {
        "prompt": [],
        "data_source": [],
        "ability": [],
        "reward_model": [],
        "extra_info": [],
        "agent_name": [],
    }
    
    # Read SWE-bench data
    if swebench_path.endswith(".json"):
        with open(swebench_path, "r") as f:
            data = json.load(f)
    elif swebench_path.endswith(".jsonl"):
        data = []
        with open(swebench_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {swebench_path}")
    
    for idx, item in enumerate(data):
        instance_id = item.get("instance_id", f"instance_{idx}")
        problem_statement = item.get("problem_statement", "")
        
        # Repository info
        repo_info = {
            "repo_name": item.get("repo", ""),
            "base_commit": item.get("base_commit", ""),
        }
        
        # Create prompt
        prompt = create_swe_prompt(problem_statement, repo_info)
        
        # Reward model config
        reward_model = {
            "style": "swe_bench",
            "instance_id": instance_id,
            "test_patch": item.get("test_patch", ""),
            "gold_patch": item.get("patch", ""),
        }
        
        # Extra info
        extra_info = {
            "index": idx,
            "split": split,
            "instance_id": instance_id,
            "repo": item.get("repo", ""),
            "base_commit": item.get("base_commit", ""),
            "problem_statement": problem_statement,
            "hints_text": item.get("hints_text", ""),
            "created_at": item.get("created_at", ""),
            "version": item.get("version", ""),
            "FAIL_TO_PASS": item.get("FAIL_TO_PASS", ""),
            "PASS_TO_PASS": item.get("PASS_TO_PASS", ""),
        }
        
        rl_dataset["prompt"].append(prompt)
        rl_dataset["data_source"].append("swe_bench_lite")
        rl_dataset["ability"].append("software_engineering")
        rl_dataset["reward_model"].append(reward_model)
        rl_dataset["extra_info"].append(extra_info)
        rl_dataset["agent_name"].append(agent_name)
    
    return pd.DataFrame(data=rl_dataset)


def main():
    parser = argparse.ArgumentParser(description="SWE Agent Dataset Generator")
    parser.add_argument("--mode", choices=["simple", "swebench"], default="simple",
                       help="Data generation mode: 'simple' for test cases, 'swebench' for SWE-bench Lite")
    parser.add_argument("--train_size", type=int, default=100, help="Number of training samples (for simple mode)")
    parser.add_argument("--test_size", type=int, default=10, help="Number of testing samples (for simple mode)")
    parser.add_argument("--swebench_train", type=str, default=None, help="Path to SWE-bench train data")
    parser.add_argument("--swebench_test", type=str, default=None, help="Path to SWE-bench test data")
    parser.add_argument("--output_dir", default="data/swe_agent", help="Directory to save the dataset")
    parser.add_argument("--agent_name", default="swe_agent", help="Name of the agent")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check pyarrow
    try:
        import pyarrow
    except ImportError:
        raise ImportError(
            "pyarrow is required for parquet support. "
            "Please install it with: pip install pyarrow"
        )
    
    if args.mode == "simple":
        # Generate simple test data
        print("Generating simple test data...")
        train_dataset = generate_simple_test_data(args.train_size, "train", args.agent_name)
        test_dataset = generate_simple_test_data(args.test_size, "test", args.agent_name)
    else:
        # Load SWE-bench data
        print("Loading SWE-bench Lite data...")
        if args.swebench_train is None or args.swebench_test is None:
            raise ValueError("--swebench_train and --swebench_test are required for swebench mode")
        train_dataset = load_swebench_lite(args.swebench_train, "train", args.agent_name)
        test_dataset = load_swebench_lite(args.swebench_test, "test", args.agent_name)
    
    # Save dataset
    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    
    print(f"\nDataset generation completed!")
    print(f"Train dataset: {len(train_dataset)} samples -> {train_path}")
    print(f"Test dataset: {len(test_dataset)} samples -> {test_path}")
    
    # Print dataset sample
    print("\n=== Sample data ===")
    print("Columns:", list(train_dataset.columns))
    if len(train_dataset) > 0:
        sample = train_dataset.iloc[0]
        print("\nFirst sample:")
        print(f"  prompt: {str(sample['prompt'])[:200]}...")
        print(f"  data_source: {sample['data_source']}")
        print(f"  agent_name: {sample['agent_name']}")


if __name__ == "__main__":
    main()
