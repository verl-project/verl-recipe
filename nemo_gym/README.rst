NVIDIA NeMo Gym Integration
==================================

`NVIDIA NeMo Gym <https://github.com/NVIDIA-NeMo/Gym>`_ is a framework for building training
and evaluation environment for multi-modal models and agentic systems. It is tested at scale, supports
multi-environment training, and provides a unified interface for training and evaluation.
Environments can be tested in NeMo Gym alone before training with verl. 
Visit the `NeMo Gym docs <https://docs.nvidia.com/nemo/gym/latest/index.html>`_
to learn more. This recipe demonstrates offline rollout collection, and single and multi-environment 
training on math and agentic workplace tasks with DAPO.

Quickstart
----------

Local Rollout Collection
~~~~~~~~~~~~~~~~~~~~~~~~

**1. Clone repositories**

Clone verl alongside NeMo Gym. The recipe lives in a submodule of verl, so
``--recurse-submodules`` (or a follow-up ``git submodule update --init``) is required.

.. code-block:: bash

    cd $WORKSPACE  # wherever you want
    git clone --recurse-submodules https://github.com/verl-project/verl.git
    git clone https://github.com/NVIDIA-NeMo/Gym.git

If you already cloned verl without submodules:

.. code-block:: bash

    cd $WORKSPACE/verl
    git submodule update --init --recursive

**2. Set up NeMo Gym for local development (note: PyPI installation is also supported)**

.. code-block:: bash

    cd $WORKSPACE/Gym

    # Install uv if needed
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env

    export UV_CACHE_DIR=/path/to/cache  # optional, useful on shared filesystems
    uv venv --python 3.12
    source .venv/bin/activate
    uv sync --extra dev

**3. Create an env.yaml with your policy model**

For standalone testing, point at a local vllm instance (or an endpoint like OpenAI):

.. code-block:: bash

    cd $WORKSPACE/Gym
    cat > env.yaml <<'EOF'
    policy_base_url: https://localhost:8000/v1
    policy_api_key: empty
    policy_model_name: Qwen/Qwen3-4B-Instruct-2507
    EOF

**4. Start servers and test an environment**

.. code-block:: bash

    cd $WORKSPACE/Gym
    source .venv/bin/activate

    config_paths="resources_servers/workplace_assistant/configs/workplace_assistant.yaml,\
    responses_api_models/vllm_model/configs/vllm_model.yaml"

    ng_run "+config_paths=[${config_paths}]"

**5. Collect and inspect rollouts**

In a separate terminal:

.. code-block:: bash

    cd $WORKSPACE/Gym
    source .venv/bin/activate

    ng_collect_rollouts \
        +agent_name=workplace_assistant_simple_agent \
        +input_jsonl_fpath=resources_servers/workplace_assistant/data/example.jsonl \
        +output_jsonl_fpath=results/rollouts.jsonl \
        +limit=5

    head -1 results/rollouts.jsonl | jq

**6. Prepare training data**

.. code-block:: bash

    cd $WORKSPACE/Gym
    source .venv/bin/activate

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

**7. Configure paths and secrets**

The submit scripts source a ``config.env`` file for secrets and paths. Copy
``config.env.example`` and fill in your values:

.. code-block:: bash

    cd $WORKSPACE/verl
    cp recipe/nemo_gym/config.env.example config.env

.. code-block:: bash

    # edit config.env
    VERL_ROOT=/path/to/verl
    NEMO_GYM_ROOT=/path/to/nemo-gym
    HF_HOME=/path/to/hf_home       # Hugging Face model cache
    RESULTS_ROOT=/path/to/results  # checkpoints and rollout dumps
    WANDB_USERNAME=your_username
    WANDB_API_KEY=your_key

**8. Launch training**

See ``submit_math.sh``, ``submit_workplace.sh``, or ``submit_multienv.sh`` for Slurm submission examples:

.. code-block:: bash

    cd $WORKSPACE/verl
    sbatch recipe/nemo_gym/submit_workplace.sh

The primary arguments relevant to NeMo Gym:

.. code-block:: bash

    +data.custom_cls.path=recipe/nemo_gym/dataset.py
    +data.custom_cls.name=NeMoGymJSONLDataset
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.nemo_gym.agent_loop.NeMoGymAgentLoopManager
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

Requirements
------------

- A NeMo Gym clone (0.2.1+) with the environments you want to train on.
- ``pip install -e $NEMO_GYM_ROOT`` in the container at job start. ``pip install nemo-gym`` also works if you don't need to modify environments. Pin to ``nemo-gym==0.2.1`` if you run into any issues on newer versions.
- Container: ``verlai/verl:vllm017.latest`` (vLLM 0.17.0).

Config YAML
-----------

Each training run needs a config YAML (see ``configs/math.yaml`` for an example):

.. code-block:: yaml

    nemo_gym:
      nemo_gym_root: $NEMO_GYM_ROOT
      uses_reasoning_parser: false          # set true for reasoning models
      config_paths:
        - $NEMO_GYM_ROOT/responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
        - $NEMO_GYM_ROOT/resources_servers/your_env/configs/your_env.yaml

Implementation Details
----------------------

- ``agent_loop.py`` — ``NeMoGymAgentLoopManager``: wraps NeMo Gym's rollout collection interface
  to collect rollouts for input tasks. Converts results to verl's DataProto format.
- ``dataset.py`` — ``NeMoGymJSONLDataset``: loads NeMo Gym JSONL datasets.
- ``replica.py`` — ``NeMoGymvLLMReplica`` / ``NeMoGymvLLMHttpServer``: verl vLLM server subclass
  that applies the patches in ``server_patch.py`` on server startup. Only instantiated when
  ``NeMoGymAgentLoopManager`` is the active agent-loop manager, so non-recipe users are unaffected.
- ``server_patch.py`` — two monkey patches applied by ``NeMoGymvLLMHttpServer``:

  - ``patch_serving_chat_for_nemo_gym()`` — patches vLLM's ``OpenAIServingChat`` and
    ``OpenAIServingTokenization`` to splice original token IDs back into the tokenized prompt,
    preventing retokenization drift in multi-step rollouts (matches NeMo RL's approach).
  - ``patch_hermes_tool_parser_thread_safety()`` — caches the tokenizer encode/decode results
    in ``Hermes2ProToolParser.__init__`` so concurrent requests don't race the Rust tokenizer
    and crash with ``RuntimeError: Already borrowed``. Based on prime-rl PR #1837.

  **Tested with vLLM 0.17.0** (``verlai/verl:vllm017.latest``). The ``_preprocess_chat`` return
  structure may change between vLLM versions — see comment in ``server_patch.py``.
