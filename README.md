# verl-recipe

`verl-recipe` hosts recipes based on [verl](https://github.com/verl-project/verl) contributed by the community.

## Usage

`verl-recipe` can be used as a submodule of `verl`, keeping backward compatibility as `verl/recipe`:

```bash
git clone https://github.com/verl-project/verl.git
cd verl
git submodule update --init --recursive recipe
```

## Required `verl` version per recipe

Every recipe directory ships a small **`REQUIRED_VERL.txt`** next to its `README.md` (same filename everywhere). That file is the canonical place for:

- upstream git URL (today [`verl-project/verl`](https://github.com/verl-project/verl), historically `volcengine/verl`),
- whether the recipe **tracks `main`** (`pip install -e .` from the same tree) or **pins** a git commit / release tag,
- a copy-pastable `pip install …` line when a pin exists.

**Rolling recipes** (`MODE=rolling` and similar): `REQUIRED_VERL.txt` now lists **exact commit IDs** taken from this workspace when the file was last refreshed:

- **`VERL_COMMIT`**: `git rev-parse HEAD` in the parent **verl** repository (core `pip install verl@git+…@VERL_COMMIT`).
- **`RECIPE_SUBMODULE_COMMIT`**: `git rev-parse HEAD` inside the **`recipe/`** submodule checkout embedded at that verl revision.
- **`RECIPE_FOLDER_LAST_COMMIT`**: `git log -1 --format=%H -- <folder>` inside **`recipe/`** for that recipe’s subdirectory only.

Together these pin both the library and the bundled recipe tree. **`REFRESH=`** at the bottom of each file is the command line used to recompute the fields after you bump verl or the submodule.

Each recipe `README.md` links to its `REQUIRED_VERL.txt` in a short **Required `verl` version** section. The repository root [`README.md`](../README.md) also points here for discoverability.

### One-shot installer: [`install_verl.sh`](install_verl.sh)

`install_verl.sh` reads a recipe's `REQUIRED_VERL.txt` and installs the pinned core `verl` for you. It understands every `MODE=` value listed above and prints the command it will run before executing it (use `--show` for a dry-run).

```bash
# From the root of this repo (the directory containing install_verl.sh):

# List every recipe + its pinned `pip install` line
./install_verl.sh --list

# Dry-run: see exactly which pip command a recipe would execute
./install_verl.sh --recipe dapo --show

# Install the pinned verl core for a recipe via pip (default)
./install_verl.sh --recipe retool

# Clone upstream verl at the pinned commit, init the recipe submodule,
# and `pip install -e .` from the checkout (useful when you want an
# editable install or plan to modify verl itself).
./install_verl.sh --recipe langgraph_agent --method git --dest ./verl

# Recipes that expose multiple variants: pick one with --option
./install_verl.sh --recipe dapo   --option reproduction   # DAPO paper SHA
./install_verl.sh --recipe flowrl --option A              # FlowRL v0.4.0 tag
./install_verl.sh --recipe spin   --option baseline       # SPIN v0.3.0.post1
```

Flags: `--recipe NAME` (e.g. `gkd/megatron`, `specRL/histoSpec`) or `--file PATH/TO/REQUIRED_VERL.txt`, `--method pip|git` (default `pip`), `--dest DIR` (default `./verl`, only used with `--method git`), `--show` (dry-run), `--yes` (skip confirmation), `--list`, `--help`.

The script requires only `bash`, `git`, `awk`, and `pip`/`pip3` on `PATH`. It does **not** `source` or `eval` the `REQUIRED_VERL.txt` file — values are extracted with `awk` and the exact command to be executed is echoed before it runs.

| Recipe | `REQUIRED_VERL.txt` |
| --- | --- |
| char_count | [`recipe/char_count/REQUIRED_VERL.txt`](char_count/REQUIRED_VERL.txt) |
| collabllm | [`recipe/collabllm/REQUIRED_VERL.txt`](collabllm/REQUIRED_VERL.txt) |
| dapo | [`recipe/dapo/REQUIRED_VERL.txt`](dapo/REQUIRED_VERL.txt) |
| deepeyes | [`recipe/deepeyes/REQUIRED_VERL.txt`](deepeyes/REQUIRED_VERL.txt) |
| entropy | [`recipe/entropy/REQUIRED_VERL.txt`](entropy/REQUIRED_VERL.txt) |
| fapo | [`recipe/fapo/REQUIRED_VERL.txt`](fapo/REQUIRED_VERL.txt) |
| fault_recover | [`recipe/fault_recover/REQUIRED_VERL.txt`](fault_recover/REQUIRED_VERL.txt) |
| flash_rl_ascend | [`recipe/flash_rl_ascend/REQUIRED_VERL.txt`](flash_rl_ascend/REQUIRED_VERL.txt) |
| flowrl | [`recipe/flowrl/REQUIRED_VERL.txt`](flowrl/REQUIRED_VERL.txt) |
| genrm_remote | [`recipe/genrm_remote/REQUIRED_VERL.txt`](genrm_remote/REQUIRED_VERL.txt) |
| gkd/megatron | [`recipe/gkd/megatron/REQUIRED_VERL.txt`](gkd/megatron/REQUIRED_VERL.txt) |
| gvpo | [`recipe/gvpo/REQUIRED_VERL.txt`](gvpo/REQUIRED_VERL.txt) |
| infigui-g1 | [`recipe/infigui-g1/REQUIRED_VERL.txt`](infigui-g1/REQUIRED_VERL.txt) |
| langgraph_agent | [`recipe/langgraph_agent/REQUIRED_VERL.txt`](langgraph_agent/REQUIRED_VERL.txt) |
| minicpmo | [`recipe/minicpmo/REQUIRED_VERL.txt`](minicpmo/REQUIRED_VERL.txt) |
| open_math_reasoning | [`recipe/open_math_reasoning/REQUIRED_VERL.txt`](open_math_reasoning/REQUIRED_VERL.txt) |
| prime | [`recipe/prime/REQUIRED_VERL.txt`](prime/REQUIRED_VERL.txt) |
| qat | [`recipe/qat/REQUIRED_VERL.txt`](qat/REQUIRED_VERL.txt) |
| r1 | [`recipe/r1/REQUIRED_VERL.txt`](r1/REQUIRED_VERL.txt) |
| r1_ascend | [`recipe/r1_ascend/REQUIRED_VERL.txt`](r1_ascend/REQUIRED_VERL.txt) |
| rep_exp | [`recipe/rep_exp/REQUIRED_VERL.txt`](rep_exp/REQUIRED_VERL.txt) |
| retool | [`recipe/retool/REQUIRED_VERL.txt`](retool/REQUIRED_VERL.txt) |
| specRL/histoSpec | [`recipe/specRL/histoSpec/REQUIRED_VERL.txt`](specRL/histoSpec/REQUIRED_VERL.txt) |
| spin | [`recipe/spin/REQUIRED_VERL.txt`](spin/REQUIRED_VERL.txt) |
| spo | [`recipe/spo/REQUIRED_VERL.txt`](spo/REQUIRED_VERL.txt) |
| sppo | [`recipe/sppo/REQUIRED_VERL.txt`](sppo/REQUIRED_VERL.txt) |
| swe_agent | [`recipe/swe_agent/REQUIRED_VERL.txt`](swe_agent/REQUIRED_VERL.txt) |

## Available Recipes (high level)

- [retool](https://github.com/verl-project/verl-recipe/tree/main/retool): Reinforcement Learning for Strategic Tool Use in LLMs
- [langgraph_agent](https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent): A tiny example to demonstrate multi-turn rollout with [LangGraph ReactAgent](https://langchain-ai.github.io/langgraph/agents/overview/) to solve math expression.
- [spo](https://github.com/verl-project/verl-recipe/tree/main/spo): [Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232).
- TBA...

## Contribution

### Version Specification

Add or update **`REQUIRED_VERL.txt`** whenever a recipe gains a new tested pin or intentionally moves forward on `main`. Examples of valid `pip` forms:

```
# release version
verl==0.6.0

# dev version
verl@git+https://github.com/verl-project/verl.git@313dfdb2199124a37189e32e6d4a6c654379f2d4
```

### Code Linting and Formatting

To maximize flexiblility but minimize meaningless changes, we apply `pre-commit` but only force code linting and formatting with `ruff`. Use it as follows:

```bash
pip install pre-commit
pre-commit install
# for staged changes
pre-commit run
# for all files in the repo
pre-commit run --all-files
```
