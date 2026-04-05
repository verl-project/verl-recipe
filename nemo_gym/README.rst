NVIDIA NeMo Gym Integration
==================================

`NVIDIA NeMo Gym <https://github.com/NVIDIA-NeMo/Gym>`_ is an RL environment framework for
scalable, multi-environment, and agentic RL. Environments can be tested in NeMo Gym alone before
training with verl. Visit the `NeMo Gym docs <https://docs.nvidia.com/nemo/gym/latest/index.html>`_
to learn more. This recipe demonstrates offline rollout collection, and single and multi-environment 
training on math and agentic workplace tasks with DAPO.

Quickstart
----------

Local Rollout Collection
~~~~~~~~~~~~~~~~~~~~~~~~

**1. Clone repositories**

.. code-block:: bash

    git clone https://github.com/verl-project/verl.git
    git clone https://github.com/NVIDIA-NeMo/Gym.git
    cd Gym

**2. Set up NeMo Gym**

.. code-block:: bash

    # Install uv if needed
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    export UV_CACHE_DIR=/path/to/cache  # optional, useful on some
    uv venv --python 3.12
    source .venv/bin/activate
    uv sync --extra dev

**3. Create an env.yaml with your policy model**

For standalone testing, point at a local vllm instance (or an endpoint like OpenAI):

.. code-block:: yaml

    # env.yaml
    policy_base_url: https://localhost:8000/v1
    policy_api_key: empty
    policy_model_name: Qwen/Qwen3-4B-Instruct-2507

**4. Start servers and test an environment**

.. code-block:: bash

    config_paths="resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
    responses_api_models/vllm_model/configs/vllm_model.yaml"

    ng_run "+config_paths=[${config_paths}]"

**5. Collect and inspect rollouts**

In a separate terminal:

.. code-block:: bash

    ng_collect_rollouts \
        +agent_name=workplace_assistant_simple_agent \
        +input_jsonl_fpath=resources_servers/workplace_assistant/data/example.jsonl \
        +output_jsonl_fpath=results/rollouts.jsonl \
        +limit=5

    head -1 results/rollouts.jsonl | jq

**6. Prepare training data**

.. code-block:: bash

    config_paths="resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
    responses_api_models/vllm_model/configs/vllm_model_for_training.yaml"

    ng_prepare_data \
        "+config_paths=[${config_paths}]" \
        +output_dirpath=data/workplace_assistant \
        +mode=train_preparation \
        +should_download=true \
        +data_source=huggingface

Check that each row has an ``agent_ref`` field. This is required for training.

Training
~~~~~~~~

**7. Launch training**

See ``submit_math.sh``, ``submit_workplace.sh``, or ``submit_multienv.sh`` for Slurm submission examples. The primary arguments relevant to NeMo Gym:

.. code-block:: bash

    +data.custom_cls.path=recipe/nemo_gym/dataset.py
    +data.custom_cls.name=NemoGymJSONLDataset
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.nemo_gym.agent_loop.NemoGymAgentLoopManager
    +actor_rollout_ref.rollout.agent.agent_loop_config_path=/path/to/configs/workplace.yaml

Multi-Environment Training
--------------------------

To train on multiple environments simultaneously, create a mixed dataset where each row has an
``agent_ref`` pointing to its environment, and include all environment config paths:

.. code-block:: yaml

    # configs/multienv.yaml
    nemo_gym:
      nemo_gym_root: $NEMO_GYM_ROOT
      config_paths:
        - $NEMO_GYM_ROOT/responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
        - $NEMO_GYM_ROOT/resources_servers/math_with_judge/configs/math_with_judge.yaml
        - $NEMO_GYM_ROOT/resources_servers/workplace_assistant/configs/workplace_assistant.yaml

The first config tells verl and nemo gym to launch the model server, which tracks token IDs and log probs to prevent
retokenization mismatches and standardizes generation behind the OpenAI Responses API.

The remaining configs define the environment. Each config specifies an agent server and optionally
a resources server that provides tools, state, verification, and reward logic. Some environments
use a ``responses_api_agents`` server only and do not have a separate resources server.

The data blend determines the sampling ratio between environments. If environment curriculum or
precise blending is desired, do not shuffle the dataset after creation. NeMo Gym routes each row
to its environment via the ``agent_ref`` field.

Note that some NeMo Gym environments such as SWE-RL launch containers and may require additional
setup (e.g. Apptainer). See each environment's README in the NeMo Gym repo for details.

Overview
--------

- ``agent_loop.py`` — ``NemoGymAgentLoopManager``: wraps NeMo Gym's rollout collection interface
  to collect rollouts for input tasks. Converts results to verl's DataProto format.
- ``dataset.py`` — ``NemoGymJSONLDataset``: loads NeMo Gym JSONL datasets.
- ``server_patch.py`` — patches vLLM's ``OpenAIServingChat`` and ``OpenAIServingTokenization``
  to correct for retokenization errors in multi-step rollouts, matching NeMo RL's approach.
  **Tested with vLLM 0.17.0** (``verlai/verl:vllm017.latest``). The ``_preprocess_chat`` return
  structure may change between vLLM versions — see comment in ``server_patch.py``.

Requirements
------------

- A NeMo Gym clone with the environments you want to train on.
- ``pip install -e /path/to/nemo-gym`` in the container at job start.
- Container: ``verlai/verl:vllm017.latest`` (vLLM 0.17.0).

Environment Variables
---------------------

The submit scripts source a ``config.env`` file for secrets and paths. Copy
``config.env.example`` and fill in your values:

.. code-block:: bash

    cp recipe/nemo_gym/config.env.example config.env

.. code-block:: bash

    VERL_ROOT=/path/to/verl
    NEMO_GYM_ROOT=/path/to/nemo-gym
    HF_HOME=/path/to/hf_home       # Hugging Face model cache
    RESULTS_ROOT=/path/to/results  # checkpoints and rollout dumps
    WANDB_USERNAME=your_username
    WANDB_API_KEY=your_key

Config YAML
-----------

Each training run needs a config YAML (see ``configs/math.yaml`` for an example):

.. code-block:: yaml

    nemo_gym:
      nemo_gym_root: $NEMO_GYM_ROOT        # path to NeMo Gym clone, expanded at runtime
      uses_reasoning_parser: false          # set true for reasoning models (e.g. DeepSeek-R1)
      config_paths:
        - $NEMO_GYM_ROOT/responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
        - $NEMO_GYM_ROOT/resources_servers/your_env/configs/your_env.yaml
