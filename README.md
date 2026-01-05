# verl-recipe

`verl-recipe` hosts recipes based on [verl](https://github.com/volcengine/verl) contributed by the community.

## Contribution

### Version Specification

Recipes are recommended to specify the verl version required, e.g.,

```
# release version
verl==0.6.0

# dev version
verl@git+https://github.com/volcengine/verl.git@313dfdb2199124a37189e32e6d4a6c654379f2d4
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

## Available Recipes

- [retool](https://github.com/verl-project/verl-recipe/tree/main/retool): Reinforcement Learning for Strategic Tool Use in LLMs
- [langgraph_agent](https://github.com/verl-project/verl-recipe/tree/main/langgraph_agent): A tiny example to demonstrate multi-turn rollout with [LangGraph ReactAgent](https://langchain-ai.github.io/langgraph/agents/overview/) to solve math expression.
- [spo](https://github.com/verl-project/verl-recipe/tree/main/spo): [Single-stream Policy Optimization](https://arxiv.org/abs/2509.13232).
- TBA...
