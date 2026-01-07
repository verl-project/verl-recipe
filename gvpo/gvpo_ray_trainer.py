import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance


class RayGVPOTrainer(RayPPOTrainer):
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        n = self.config.actor_rollout_ref.rollout.n
        bs_per_gpu = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        assert world_size * bs_per_gpu % n == 0, (
            f"GVPO requires: world_size {world_size} * bs_per_gpu {bs_per_gpu} should be divisible by n {n}"
        )
        k_partitions = batch_size // (world_size * bs_per_gpu)
        phase1_seqlen_lst = [sum(global_seqlen_lst[i : i + n]) for i in range(0, len(global_seqlen_lst), n)]
        if batch_size % (world_size * bs_per_gpu) == 0:
            phase1_partition_lst = get_seqlen_balanced_partitions(
                phase1_seqlen_lst, k_partitions=k_partitions, equal_size=True
            )
        else:
            if k_partitions > 0:
                phase1_partition_lst = get_seqlen_balanced_partitions(
                    phase1_seqlen_lst[: k_partitions * world_size * bs_per_gpu / n],
                    k_partitions=k_partitions,
                    equal_size=True,
                )
            else:
                phase1_partition_lst = []
            phase1_partition_lst.append(
                list(range(k_partitions * world_size * bs_per_gpu // n + 1, len(phase1_seqlen_lst)))
            )

        global_idx = [-1] * batch_size
        for k in range(len(phase1_partition_lst)):
            partition = phase1_partition_lst[k]
            phase2_seqlen_lst = [global_seqlen_lst[i * n + j] for i in partition for j in range(n)]
            inx = [i * n + j for i in partition for j in range(n)]
            phase2_partition_lst = get_seqlen_balanced_partitions(
                phase2_seqlen_lst, k_partitions=world_size, equal_size=True
            )
            for i in range(len(phase2_partition_lst)):
                for j in range(len(phase2_partition_lst[i])):
                    global_idx[i * (batch_size // world_size) + k * bs_per_gpu + j] = inx[phase2_partition_lst[i][j]]

        global_partition_lst = [
            global_idx[i * (batch_size // world_size) : (i + 1) * (batch_size // world_size)] for i in range(world_size)
        ]
        global_idx = torch.tensor(global_idx)

        batch.union(DataProto.from_single_dict({"uid_tensor": torch.Tensor([i // n for i in range(batch_size)])}))
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
