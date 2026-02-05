# Recipe: Tokens saving and Auto Fault Recover for Rollout

## Design

[RFC](https://github.com/volcengine/verl/discussions/4355)

## Solution

![Req Resume Performance](https://github.com/user-attachments/assets/58127f8f-c3f8-43f2-9e54-198cfd22d705)

## Support

algorithm

- [x] grpo

rollout

- [x] vllm
- [ ] sglang

train

- [x] megatron
- [ ] fsdp

## Version

```bash
# dev version
pip install verl@git+https://github.com/volcengine/verl.git@b97ebfd5062223337ae065c2250f8ab5c0e08e5e
```

## Quickstart

```bash
# refer to this example: recipe/fault_recover/run_qwen2_5_0.5b_megatron.sh
python3 -m recipe.fault_recover.main_ppo --config-path=config \
    --config-name='fault_recover_ppo_megatron_trainer.yaml' \
    fault_manager.enable=True \
    actor_rollout_ref.rollout.agent.default_agent_loop=fault_recover_single_turn_agent \
    # refer to other detail config in the fault_manager part of
    # recipe/fault_recover/config/fault_recover_ppo_megatron_trainer.yaml
```

## Configuration

```yaml
fault_manager:
  enable: False
  # max retry times for other training phases except rollout (restart ray)
  max_reschedule_times: 1
  # max retry times for rollout phase (rebuild worker group)
  max_rebuild_times: 1
  # timeout of waiting cluster to be ready
  timeout_rebuild: 300
  # check chips usage interval during rollout, set -1 to disable timeout check
  timeout_task_check_interval: 10
  # timeout of chips usage being free, set -1 to disable chip check and
  # 'timeout_task_check_interval' will be the whole time limit of rollout
  # which means you should increase it
  timeout_chip_free: 30
  # file path for token saving
  tokens_save_file: ./tokens_ckpt/tokens.pt
  # interval of saving tokens to disk, remember to clear if training config is changed 
  tokens_save_interval: 10
```

## FAQ
