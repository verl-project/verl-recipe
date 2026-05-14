#!/bin/bash

# PYTHONPATH 必须在 python 启动前生效，无法迁移到 yaml。
# 其余原本通过环境变量设置的参数已迁移为 Hydra override，
# 优先级：命令行 override > shell 环境变量 > yaml 默认值。
PYTHONPATH="/home/verl/harbor/src/harbor:$PYTHONPATH" \
python3 -m recipe.agentic.agentic_main \
    proxy_server.llm_proxy_ip=10.0.30.11 \
    remote_agent.agent_name=swe-agent \
    remote_agent.model_name=openai/qwen-max \
    remote_agent.use_local_trial=true \
    remote_agent.task_path_template='/home/verl/dataset-tasks/{instance_id}' \
    'remote_agent.agent_kwargs={total_cost_limit: 0, per_instance_cost_limit: 0}' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=['/var/model-dataset/princeton-nlp/data/train-00000-of-00001.1.parquet'] \
    data.val_files=['/var/model-dataset/princeton-nlp/data/test1-00000-of-00001.10.parquet'] \
    data.return_raw_chat=true \
    data.train_batch_size=1 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.prompt_key=instance_id \
    reward_model.use_reward_loop=true \
    actor_rollout_ref.model.path=/var/model/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24576 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=remote_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=verl/experimental/agent_loop/swe-agent.yaml \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    trainer.logger=[\"console\"] \
    trainer.project_name=remote-agent \
    trainer.experiment_name=qwen2.5-3b \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=true \
    trainer.log_val_generations=50 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.default_local_dir=/var/checkpoint/qwen2.5-7b \
    trainer.test_freq=5 \
    trainer.total_epochs=1
