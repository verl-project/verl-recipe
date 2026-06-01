## 问题分析
你的核心问题是： kernel-verifier 作为 skill 由大模型调用执行，但验证过程本身需要大模型来编排（创建文件、调用脚本、收集结果），这导致大模型可能在验证环节产生幻觉，比如：

1. 伪造验证结果 — 没有真正运行 verify.py ，却声称验证通过
2. 篡改/忽略错误信息 — 验证实际失败了，但大模型在传递结果时"美化"了错误
3. 跳过验证步骤 — 大模型可能认为代码"看起来正确"就跳过验证
4. 错误解读脚本输出 — 误读 exit code 或 stderr
从代码来看，你已经做了一些约束（如 SKILL.md 中明确写了 "⛔ 禁止事项"），但这些约束是 软约束 （prompt 层面），大模型仍然可以绕过。

## 解决方案：将验证从 Skill 提升为确定性工具（Tool）
核心思路： 把验证脚本从"由大模型编排调用的 skill"变为"确定性执行的工具"，让验证过程完全脱离大模型的控制。

### 方案一：将验证脚本封装为 Agent 的 bash 工具强制调用 + 结果硬解析
这是最小改动方案，核心改动在 triton-ascend-coder.md 的 Agent 定义中：

改动要点 ：

1. 在 Agent 工作流中，将验证步骤从 "调用 kernel-verifier skill" 改为 "直接通过 bash 执行验证脚本 + 硬性结果解析规则"
2. 验证结果不再由大模型"解读"，而是通过 exit code + 结构化 JSON 输出自动判定
具体来说，修改 triton-ascend-coder.md 中 Phase 3 的 3.2 和 3.3 步骤：

```
### 3.2 AST 预检查（确定性执行）

执行以下命令，**必须**通过 bash 工具运行：

```bash
python3 <skill_path>/scripts/validate_triton_impl.py <generated_code_path> --json
```

**结果判定规则（硬性，禁止大模型自行判断）**：
- exit code == 0 → 通过，继续 3.3
- exit code != 0 → 退化，解析 stdout 的 JSON 获取 regression_type 和 suggestion
  - **禁止**忽略非零 exit code
  - **禁止**在 exit code != 0 时继续执行 3.3

### 3.3 功能验证（确定性执行）

执行以下命令，**必须**通过 bash 工具运行：

```bash
python3 <skill_path>/scripts/verify.py \
    --op_name <op_name> \
    --verify_dir <verify_dir> \
    --triton_impl_name <impl_name> \
    --timeout 900
```

**结果判定规则（硬性，禁止大模型自行判断）**：
- exit code == 0 且 stdout 包含 "验证成功" → 验证通过
- exit code != 0 → 验证失败，verifier_error = stderr 完整内容
  - **禁止**在 exit code != 0 时声称验证通过
  - **禁止**截断或修改 stderr 内容
```
但这仍然是 prompt 层面的约束，本质上还是"软约束"。

### 方案二（推荐）：将验证脚本注册为 Agent 的确定性 Tool
这是最彻底的方案。将 verify.py 和 benchmark.py 封装为 Agent 可调用的 确定性工具 （而非 skill），工具的输出是结构化的，大模型无法篡改。

具体做法 ：

1. 创建一个 wrapper 脚本 ，将验证脚本的调用和结果解析封装为确定性的命令行工具，输出结构化 JSON：
```
# scripts/deterministic_verify.py
#!/usr/bin/env python3
"""确定性验证工具 — 封装 verify.py，输出结构化 JSON 结果。
大模型只需读取 JSON，无需解读原始输出。"""
import subprocess, json, sys

def run_verify(op_name, verify_dir, triton_impl_name, timeout=900):
    cmd = [
        sys.executable, "verify.py",
        "--op_name", op_name,
        "--verify_dir", verify_dir,
        "--triton_impl_name", triton_impl_name,
        "--timeout", str(timeout),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout+30)
        result = {
            "passed": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.decode("utf-8", errors="replace"),
            "stderr": proc.stderr.decode("utf-8", errors="replace"),
        }
    except subprocess.TimeoutExpired:
        result = {
            "passed": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"验证超时（{timeout}秒）",
        }
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)  # wrapper 本身总是成功退出，结果在 JSON 中

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_name", required=True)
    parser.add_argument("--verify_dir", required=True)
    parser.add_argument("--triton_impl_name", default="triton_ascend_impl")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    run_verify(args.op_name, args.verify_dir, args.triton_impl_name, args.timeout)
```
2. 修改 Agent 定义 ，将 kernel-verifier 从 skill 列表中移除，改为在 Agent 的 tools 中注册自定义 bash 命令模式：
在 triton-ascend-coder.md 中：

```
skills:
  - op-task-extractor
  - kernel-designer
  - kernel-generator
  # - kernel-verifier  ← 移除，验证不再作为 skill
  - latency-optimizer
```
3. 在 Agent 工作流中，用确定性脚本调用替代 skill 调用 ：
```
### 3.3 功能验证（确定性执行，禁止大模型解读原始输出）

执行确定性验证脚本：

```bash
python3 <skill_path>/scripts/deterministic_verify.py \
    --op_name <op_name> \
    --verify_dir <verify_dir> \
    --triton_impl_name <impl_name> \
    --timeout 900
```

**读取输出的 JSON 结果**，按以下规则判定（硬性规则，无自由裁量权）：
- `passed == true` → 验证通过
- `passed == false` → 验证失败，`stderr` 字段即为错误信息

**禁止事项**：
- 禁止在 `passed == false` 时继续后续步骤
- 禁止修改或截断 `stderr` 内容
- 禁止自行编写测试代码替代此脚本
```
### 方案三（最彻底）：引入外部验证编排器
将验证的编排逻辑从大模型完全剥离，用一个独立的 Python 脚本（验证编排器）来控制整个验证流程：

```
# scripts/verify_orchestrator.py
#!/usr/bin/env python3
"""验证编排器 — 独立于大模型，确定性执行验证流程。
大模型只需调用此脚本并读取 JSON 结果，无法干预验证过程。"""
import argparse, json, subprocess, sys, os

def run_step(cmd, timeout=None):
    """执行一个步骤，返回结构化结果"""
    try:
        proc = subprocess.run(
            cmd, capture_output=True, 
            timeout=timeout, cwd=os.getcwd()
        )
        return {
            "success": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout.decode("utf-8", errors="replace"),
            "stderr": proc.stderr.decode("utf-8", errors="replace"),
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "exit_code": -1, "stdout": "", "stderr": "超时"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_name", required=True)
    parser.add_argument("--verify_dir", required=True)
    parser.add_argument("--triton_impl_name", default="triton_ascend_impl")
    parser.add_argument("--generated_code", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip_benchmark", action="store_true")
    parser.add_argument("--output", required=True, help="结果输出路径")
    args = parser.parse_args()

    result = {"steps": []}

    # Step 1: AST 预检查
    step1 = run_step([
        sys.executable, 
        os.path.join(os.path.dirname(__file__), "validate_triton_impl.py"),
        args.generated_code, "--json"
    ])
    result["steps"].append({"name": "ast_check", **step1})
    if not step1["success"]:
        result["overall"] = "failed"
        result["failure_step"] = "ast_check"
        result["error"] = step1["stdout"]  # JSON 格式的错误信息
        with open(args.output, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        sys.exit(1)

    # Step 2: 功能验证
    step2 = run_step([
        sys.executable,
        os.path.join(os.path.dirname(__file__), "verify.py"),
        "--op_name", args.op_name,
        "--verify_dir", args.verify_dir,
        "--triton_impl_name", args.triton_impl_name,
        "--timeout", str(args.timeout),
    ])
    result["steps"].append({"name": "verify", **step2})
    if not step2["success"]:
        result["overall"] = "failed"
        result["failure_step"] = "verify"
        result["error"] = step2["stderr"]
        with open(args.output, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        sys.exit(1)

    # Step 3: 性能测试（可选）
    if not args.skip_benchmark:
        perf_output = args.output.replace(".json", "_perf.json")
        step3 = run_step([
            sys.executable,
            os.path.join(os.path.dirname(__file__), "benchmark.py"),
            "--op_name", args.op_name,
            "--verify_dir", args.verify_dir,
            "--triton_impl_name", args.triton_impl_name,
            "--output", perf_output,
        ])
        result["steps"].append({"name": "benchmark", **step3})
        if step3["success"] and os.path.exists(perf_output):
            with open(perf_output) as f:
                result["perf_data"] = json.load(f)

    result["overall"] = "passed"
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    sys.exit(0)

if __name__ == "__main__":
    main()
```
这样大模型只需要执行一条命令并读取一个 JSON 文件， 完全无法干预验证过程 。