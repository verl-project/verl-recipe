# verl-recipe

`verl-recipe` is a set of examples based on [verl](https://github.com/volcengine/verl) for end-to-end RL training recipes.

## Contributing

> [!NOTE]
> Recipes from the main `verl` repository are temperarily placed in the [`legacy`](legacy) directory since they are broken or fail to meet the following requirements.
> The original contributors are expected to fix the issues before moving them to the root directory.

### Recipe Folder Structure

All recipe should follow the following structure:

- `README.md`: recipe description
- `code`: recipe code
- `script`: reproducible training script

Specifically, `README.md` should contain following sections:

- Installation: which verl version is required for this recipe?

```
# release version
pip install verl==0.6.0

# dev version
pip install verl@git+https://github.com/volcengine/verl.git@313dfdb2199124a37189e32e6d4a6c654379f2d4
```

- Training: how to train
- Evaluation: performance metrics
- Citation: paper, notion, blog, etc.

### Code Linting and Formatting

We rely on pre-commit to keep our code consistent. To set it up:

```bash
pip install pre-commit
pre-commit install
# for staged changes
pre-commit run
# for all files in the repo
pre-commit run --all-files
# run a specific hook with pre-commit
# pre-commit run --all-files --show-diff-on-failure --color=always <hood-id>
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

## Available Recipes

- [retool](https://github.com/verl-project/verl-recipe/tree/main/retool): Reinforcement Learning for Strategic Tool Use in LLMs
- [langgraph_agent](https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent): A tiny example to demonstrate multi-turn rollout with [LangGraph ReactAgent](https://langchain-ai.github.io/langgraph/agents/overview/) to solve math expression.
- [spo](https://github.com/verl-project/verl-recipe/tree/main/spo): [Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232).
