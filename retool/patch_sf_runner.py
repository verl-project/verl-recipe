#!/usr/bin/env python3
"""Patch SandboxFusion major.py to use system python instead of conda."""
import sys
import os

sf_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/SandboxFusion"
major_py = os.path.join(sf_dir, "sandbox/runners/major.py")

with open(major_py, "r") as f:
    lines = f.readlines()

# Find and replace get_python_rt_env function
new_lines = []
skip = False
for i, line in enumerate(lines):
    if "def get_python_rt_env(env_name: str):" in line:
        # Replace the entire function
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(f"{indent}def get_python_rt_env(env_name: str):\n")
        new_lines.append(f"{indent}    # Patched: use system python directly (no conda)\n")
        new_lines.append(f"{indent}    import shutil\n")
        new_lines.append(f'{indent}    python_path = os.path.dirname(shutil.which("python3") or "/usr/bin/python3")\n')
        new_lines.append(f'{indent}    filtered_path = os.environ.get("PATH", "")\n')
        new_lines.append(f'{indent}    return {{"PATH": f"{{python_path}}:{{filtered_path}}"}}\n')
        skip = True
        continue
    if skip:
        # Skip old function body until next function or class or blank line at same indent
        stripped = line.strip()
        if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("@"):
            skip = False
            new_lines.append(line)
        elif line.strip() == "" and i + 1 < len(lines) and not lines[i+1].startswith("    "):
            skip = False
            new_lines.append(line)
        # else: skip this line (still in old function body)
        continue
    new_lines.append(line)

with open(major_py, "w") as f:
    f.writelines(new_lines)

print(f"Patched {major_py}")

# Also create missing dirs
os.makedirs(os.path.join(sf_dir, "docs/build"), exist_ok=True)
os.makedirs(os.path.join(sf_dir, "sandbox/pages"), exist_ok=True)
print("Created docs/build and sandbox/pages")
