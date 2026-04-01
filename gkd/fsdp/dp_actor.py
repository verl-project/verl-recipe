import logging
import os

import torch
from torch import nn

from verl import DataProto
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.actor.dp_actor import DataParallelPPOActor

if torch.cuda.is_available():
    pass
else:
    pass


__all__ = ["DataParallelOPKDActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def merge_teacher_student_indices_single(t_indices_list, s_indices_list, topk):
    """
    Merge teacher and student top-k indices into a unified list.

    Args:
        t_indices_list: Python list shaped like [B][R][K] for teacher top-k indices.
        s_indices_list: Python list shaped like [B][R][K] for student top-k indices.
        topk: Final number of merged indices per token position (K).

    Returns:
        merged_indices: Python list [B][R][K] with teacher-first, then student fill, no duplicates.
        overlap_counts: Per-token overlap counts between teacher and student sets, shaped [B][R].
        overlap_ratios: Per-token overlap ratios (overlap / K), shaped [B][R].
    """
    merged_indices, overlap_counts, overlap_ratios = [], [], []
    for i in range(len(t_indices_list)):
        merged_indice, overlap_count, overlap_ratio = [], [], []
        for j in range(len(t_indices_list[i])):
            t_indices = t_indices_list[i][j]
            s_indices = s_indices_list[i][j] if s_indices_list is not None else []
            set_t, set_s = set(t_indices), set(s_indices)
            n_overlap = len(set_t & set_s)
            overlap_count.append(n_overlap)
            overlap_ratio.append(n_overlap / topk if topk else 0.0)

            merged, seen, t_ptr, s_ptr = [], set(), 0, 0
            while len(merged) < topk:
                t_idx = t_indices[t_ptr] if t_ptr < len(t_indices) else None
                s_idx = s_indices[s_ptr] if s_ptr < len(s_indices) else None
                t_ptr += t_idx is not None
                s_ptr += s_idx is not None
                for idx in (t_idx, s_idx):
                    if idx is not None and idx not in seen:
                        merged.append(idx)
                        seen.add(idx)
                    if len(merged) >= topk:
                        break
            merged_indice.append(merged[:topk])
        merged_indices.append(merged_indice)
        overlap_counts.append(overlap_count)
        overlap_ratios.append(overlap_ratio)
    return merged_indices, overlap_counts, overlap_ratios


class DataParallelOPKDActor(DataParallelPPOActor):
    """
    Data-parallel actor for OPKD.

    Contract (unchanged):
      - compute_student_index(...) -> student top-k indices [B, R, K]
      - compute_union_logits(...)  -> (merged_indices [B, R, K], merged_logits [B, R, K])
        NOTE: merged_logits carries TEACHER LOGITS on the union top-k (not probabilities).
      - update_policy(...)         -> computes weighted bidirectional KL (JSD-style) and updates the student.
    """

    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config, actor_module, actor_optimizer)
        # Top-k for union subset.
        self.topk = config.get("topk", 256)

        # JSD-style alpha in [0, 1]; used for α-JSD / skewed mixture.
        self.jsd_alpha = float(config.get("jsd_alpha", 0.5))
        if not (0.0 <= self.jsd_alpha <= 1.0):
            self.jsd_alpha = min(1.0, max(0.0, self.jsd_alpha))
        # KD loss type: "forward_kl", "reverse_kl", or "jsd"
        self.kd_loss_type = str(config.get("kd_loss_type", "forward_kl")).lower()
        # Sentence-level power sampling exponent α
        self.use_power_weighting = bool(config.get("use_power_weighting", True))
        print(f"[DataParallelOPKDActor] use_power_weighting={self.use_power_weighting}")
        self.power_alpha = float(config.get("power_alpha", 1.0))

        self.coverage_coef = float(config.get("coverage_coef", 1.0))
        print(
            f"[DataParallelOPKDActor] Config: topk={self.topk}, "
            f"jsd_alpha={self.jsd_alpha}, kd_loss_type={self.kd_loss_type}, "
            f"power_alpha={self.power_alpha}, coverage_coef={self.coverage_coef}"
        )

    def _forward_micro_batch_teacher(self, micro_batch, temperature, calculate_entropy=False):
        """
        Teacher forward pass on a micro-batch.

        Returns:
            merged_indices: LongTensor [B, R, K], union top-k per response token.
            merged_logits : FloatTensor [B, R, K], TEACHER logits gathered on union indices.
            teacher_token_logp: FloatTensor [B, R], per-token log p_teacher(y|x) on the true response tokens.
        """
        response_length = micro_batch["responses"].size(-1)
        responses = micro_batch["responses"]  # [B, R]
        input_ids, attention_mask, position_ids = (
            micro_batch["input_ids"],  # [B, S]
            micro_batch["attention_mask"],  # [B, S]
            micro_batch["position_ids"],  # [B, S] (or [*, *, *] then transposed below)
        )
        student_topk_index = micro_batch["student_topk_index"]  # [B, R, K]

        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

            # output.logits: [B, S, V] -> response span logits: [B, R, V], then temperature scaling.
            logits = output.logits
            logits = logits[:, -response_length - 1 : -1, :].div(temperature)  # [B, R, V]

            # Teacher top-k directly on logits (saves memory vs. softmax).
            k = student_topk_index.shape[-1]
            logits_topk_indices = torch.topk(logits, k, dim=-1).indices  # [B, R, K]

            # Merge teacher & student sets.
            merged_indices, _, _ = merge_teacher_student_indices_single(
                logits_topk_indices.tolist(), student_topk_index.tolist(), k
            )
            merged_indices = torch.tensor(
                merged_indices, dtype=torch.int64, device=student_topk_index.device
            )  # [B, R, K]

            # Gather teacher logits on the union indices.
            merged_logits = torch.gather(logits, dim=-1, index=merged_indices)  # [B, R, K]

            # Per-token log p_teacher(y|x) on the true response tokens.
            teacher_token_logp = logprobs_from_logits(
                logits=logits,  # [B, R, V]
                labels=responses,  # [B, R]
            ).to(logits.dtype)  # [B, R]

        return merged_indices, merged_logits, teacher_token_logp

    def _forward_micro_batch_student(self, micro_batch, temperature, calculate_entropy=False):
        """
        Student forward pass on a micro-batch, returning full-vocab logits for the response span.

        Returns:
            logits: FloatTensor [B, R, V], temperature-scaled student logits on the response span.
        """
        response_length = micro_batch["responses"].size(-1)
        input_ids, attention_mask, position_ids = (
            micro_batch["input_ids"],  # [B, S]
            micro_batch["attention_mask"],  # [B, S]
            micro_batch["position_ids"],  # [B, S] (or [*, *, *] then transposed below)
        )

        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

            logits = output.logits  # [B, S, V]
            logits = logits[:, -response_length - 1 : -1, :].div(temperature)  # [B, R, V]

        return logits

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_union_logits(self, data: DataProto, calculate_entropy=False):
        """
        Compute union indices and TEACHER logits on those indices.

        Note: returned merged_logits are logits on union top-k (not probabilities).
        """
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "student_topk_index"]
        data = data.select(batch_keys=select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        merged_indices_lst, merged_logits_lst, teacher_logp_lst = [], [], []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            with torch.no_grad():
                merged_indices, merged_logits, teacher_token_logp = self._forward_micro_batch_teacher(
                    micro_batch.batch, temperature=temperature, calculate_entropy=calculate_entropy
                )
            merged_indices_lst.append(merged_indices)  # each: [b, R, K]
            merged_logits_lst.append(merged_logits)  # each: [b, R, K]
            teacher_logp_lst.append(teacher_token_logp)  # each: [b, R]

        # Concatenate all micro-batches.
        merged_indices = torch.cat(merged_indices_lst, dim=0)  # [B, R, K]
        merged_logits = torch.cat(merged_logits_lst, dim=0)  # [B, R, K]
        teacher_token_logp = torch.cat(teacher_logp_lst, dim=0)  # [B, R]

        if use_dynamic_bsz:
            merged_indices = restore_dynamic_batch(merged_indices, batch_idx_list)
            merged_logits = restore_dynamic_batch(merged_logits, batch_idx_list)
            teacher_token_logp = restore_dynamic_batch(teacher_token_logp, batch_idx_list)

        return merged_indices, merged_logits, teacher_token_logp

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_student_index(self, data: DataProto, calculate_entropy=False):
        """
        Compute student top-k indices on the response span.

        Top-k selection is performed directly on logits without building full-vocab probabilities.
        """
        self.actor_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        data = data.select(batch_keys=select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        student_indices_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            with torch.no_grad():
                logits = self._forward_micro_batch_student(
                    micro_batch.batch, temperature=temperature, calculate_entropy=calculate_entropy
                )  # [b, R, V]
                _, student_indices = torch.topk(logits, self.topk, dim=-1)  # [b, R, K]
            student_indices_lst.append(student_indices)

        student_indices = torch.cat(student_indices_lst, dim=0)  # [B, R, K]

        if use_dynamic_bsz:
            student_indices = restore_dynamic_batch(student_indices, batch_idx_list)

        return student_indices

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """
        Update the student policy with:
          - token-level KD loss on union-topk (forward KL / reverse KL / α-JSD via kd_loss_type)
          - per-prompt power-weighting across n rollouts (sentence-level signal).

        Assumption (from Ray fit + actor config):
          - rollout.n > 1
          - repeat(interleave=True) in fit()
          - ppo_mini_batch_size == ppo_micro_batch_size_per_gpu == rollout.n
          => each micro_batch contains n responses for a SINGLE prompt.
        """
        self.actor_module.train()
        temperature = data.meta_info["temperature"]

        # Allow per-batch override of α and kd_loss_type via meta_info.
        jsd_alpha = float(data.meta_info.get("jsd_alpha", self.jsd_alpha))
        jsd_alpha = max(0.0, min(1.0, jsd_alpha))
        kd_loss_type = str(data.meta_info.get("kd_loss_type", self.kd_loss_type)).lower()

        select_keys = [
            "responses",  # [B, R]
            "response_mask",  # [B, R]
            "input_ids",  # [B, S]
            "attention_mask",  # [B, S]
            "position_ids",  # [B, S]
            "merged_indices",  # [B, R, K]
            "merged_logits",  # [B, R, K] teacher logits on union indices
            "teacher_token_logp",  # [B, R]
        ]
        data = data.select(batch_keys=select_keys)

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        metrics = {}

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                # Recommend use_dynamic_bsz=False so that each micro_batch corresponds to one prompt's n rollouts.
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                self.actor_optimizer.zero_grad()

                import hashlib

                def _rank_info():
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        rank = torch.distributed.get_rank()
                        world = torch.distributed.get_world_size()
                    else:
                        rank, world = 0, 1
                    dev = torch.cuda.current_device() if torch.cuda.is_available() else -1
                    return rank, world, dev

                def _hash_prompt(input_ids, attention_mask, response_mask):
                    """
                    input_ids:      [B, S] (prompt + response padded)
                    attention_mask: [B, S]
                    response_mask:  [B, R] (mask over response span; 1=valid response token, 0=response padding)

                    return:
                        hashes: list[str] length B, short hash for prompt tokens
                        prompt_len: list[int] length B, estimated prompt lengths
                        resp_len: list[int] length B, effective response lengths (non-pad) from response_mask
                    """
                    # Estimate prompt effective length: total_effective_len - response_effective_len.
                    total_len = attention_mask.sum(dim=-1).to(torch.long)  # [B]
                    resp_len = response_mask.sum(dim=-1).to(torch.long)  # [B]
                    prompt_len = (total_len - resp_len).clamp(min=0)  # [B]

                    hashes = []
                    for b in range(input_ids.size(0)):
                        plen = int(prompt_len[b].item())
                        toks = input_ids[b, :plen].detach().cpu().to(torch.int32).numpy().tobytes()
                        h = hashlib.sha1(toks).hexdigest()[:10]
                        hashes.append(h)
                    return hashes, prompt_len.detach().cpu().tolist(), resp_len.detach().cpu().tolist()

                rank, world, dev = _rank_info()
                print(
                    f"[DBG][rank {rank}/{world}][cuda:{dev}] micro_batches={len(micro_batches)} "
                    f"mini_batch_size={len(mini_batch.batch)} micro_bsz={self.config.ppo_micro_batch_size_per_gpu}"
                )
                uids = None
                try:
                    uids = mini_batch.non_tensor_batch.get("uid", None)
                except Exception:
                    uids = None

                if uids is not None:
                    uid_list = uids.tolist()
                    total = len(uid_list)

                    from collections import Counter

                    cnt = Counter(uid_list)
                    uniq = list(cnt.keys())
                    n_unique = len(uniq)

                    print(f"[DBG][rank {rank}][cuda:{dev}] mini_batch uid summary: total={total}, unique={n_unique}")
                    uid_cnt_preview = {k: cnt[k] for k in uniq[:5]}
                    print(f"[DBG][rank {rank}][cuda:{dev}] mini_batch uid counts (preview): {uid_cnt_preview}")

                # Check whether prompts are identical inside a micro-batch (should be, if grouped by rollouts).
                for i, mb in enumerate(micro_batches[: min(4, len(micro_batches))]):  # show up to 4
                    try:
                        inp = mb.batch["input_ids"]
                        am = mb.batch["attention_mask"]
                        rm = mb.batch["response_mask"]
                        hs, plen, rlen = _hash_prompt(inp, am, rm)
                        uniq = len(set(hs))
                        print(
                            f"[DBG][rank {rank}][cuda:{dev}] micro[{i}] B={inp.size(0)} "
                            f"prompt_hash_unique={uniq} prompt_len={plen} resp_len={rlen} hashes={hs}"
                        )
                    except Exception as e:
                        print(f"[DBG][rank {rank}][cuda:{dev}] micro[{i}] prompt check failed: {repr(e)}")

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch}

                    response_mask = model_inputs["response_mask"]  # [B, R]
                    merged_indices = model_inputs["merged_indices"]  # [B, R, K]
                    teacher_sel_logits = model_inputs["merged_logits"]  # [B, R, K]
                    teacher_token_logp = model_inputs["teacher_token_logp"]  # [B, R]

                    # ---- student forward (full vocab on response span) ----
                    student_full_logits = self._forward_micro_batch_student(
                        model_inputs, temperature=temperature, calculate_entropy=False
                    )  # [B, R, V]

                    # Gather student logits on union indices.
                    student_sel_logits = torch.gather(student_full_logits, dim=-1, index=merged_indices)  # [B, R, K]

                    # ---- truncated log-softmax on union subset ----
                    t_logZ_sub = torch.logsumexp(teacher_sel_logits, dim=-1, keepdim=True)  # [B, R, 1]
                    s_logZ_sub = torch.logsumexp(student_sel_logits, dim=-1, keepdim=True)  # [B, R, 1]
                    t_logp_sub = teacher_sel_logits - t_logZ_sub  # [B, R, K]
                    s_logp_sub = student_sel_logits - s_logZ_sub  # [B, R, K]
                    t_p_sub = torch.exp(t_logp_sub)  # [B, R, K]
                    s_p_sub = torch.exp(s_logp_sub)  # [B, R, K]

                    # ---- basic KLs ----
                    # reverse KL: KL(q || p)
                    kl_reverse = (s_p_sub * (s_logp_sub - t_logp_sub)).sum(dim=-1)  # [B, R]
                    # forward KL: KL(p || q)
                    kl_forward = (t_p_sub * (t_logp_sub - s_logp_sub)).sum(dim=-1)  # [B, R]

                    # ---- true α-JSD (for metrics & "jsd" branch) ----
                    # m = α p + (1-α) q on the union subset
                    m_p_sub = jsd_alpha * t_p_sub + (1.0 - jsd_alpha) * s_p_sub  # [B, R, K]
                    m_logp_sub = torch.log(m_p_sub + 1e-12)  # [B, R, K]

                    kl_p_m = (t_p_sub * (t_logp_sub - m_logp_sub)).sum(dim=-1)  # [B, R]
                    kl_q_m = (s_p_sub * (s_logp_sub - m_logp_sub)).sum(dim=-1)  # [B, R]
                    jsd_tok = jsd_alpha * kl_p_m + (1.0 - jsd_alpha) * kl_q_m  # [B, R]

                    # ---- choose KD loss by kd_loss_type ----
                    if kd_loss_type == "forward_kl":
                        loss_tok = kl_forward  # [B, R]
                    elif kd_loss_type == "reverse_kl":
                        loss_tok = kl_reverse  # [B, R]
                    elif kd_loss_type == "jsd":
                        loss_tok = jsd_tok  # [B, R]
                    else:
                        raise ValueError(f"Unknown kd_loss_type: {kd_loss_type}")

                    # ---- (optional) entropy on union subset for diagnostics ----
                    entropy_student_tok = -(s_p_sub * s_logp_sub).sum(dim=-1)  # [B, R]
                    entropy_teacher_tok = -(t_p_sub * t_logp_sub).sum(dim=-1)  # [B, R]

                    # ---- Coverage Loss (Mass Leakage Penalty) ----
                    # Penalize probability mass leaking outside the union set.
                    # L_cov = -log(sum_{k in U} P_s(k)) = -(logZ_sub - logZ_global)
                    s_logZ_global = torch.logsumexp(student_full_logits, dim=-1, keepdim=True).squeeze(-1)  # [B, R]
                    s_logZ_sub_sq = s_logZ_sub.squeeze(-1)  # [B, R]
                    loss_coverage = -(s_logZ_sub_sq - s_logZ_global)  # [B, R] >= 0
                    loss_tok = loss_tok + self.coverage_coef * loss_coverage

                    # ==========================================================
                    # >>> per-prompt power weighting across n rollouts <<<
                    lengths = response_mask.sum(dim=-1).clamp(min=1)  # [B]

                    # sequence-level teacher confidence score: average teacher logprob per token
                    score = (teacher_token_logp * response_mask).sum(dim=-1) / lengths  # [B]

                    B = response_mask.size(0)

                    if self.use_power_weighting:
                        s = self.power_alpha * score
                        s = s - s.mean()  # numerical stability
                        w = torch.softmax(s, dim=0).detach()  # [B], stop-gradient weights
                    else:
                        # uniform weights (normalized for consistency)
                        w = torch.ones(B, device=score.device, dtype=score.dtype)

                    # normalize weights so that sum_i w_i = 1 (also handles the uniform branch)
                    w = w / w.sum().clamp(min=1)

                    # rollout-level (sequence-level) averaging: loss_i = (1/|y_i|) * sum_t loss_tok(i,t)
                    loss_i = (loss_tok * response_mask).sum(dim=-1) / lengths  # [B]

                    # teacher-confidence-weighted rollout average (matches Eq. (method_prompt_obj))
                    loss = (w * loss_i).sum()
                    # <<< per-prompt power weighting <<<
                    # ==========================================================

                    if not self.config.use_dynamic_bsz:
                        loss = loss / self.gradient_accumulation

                    loss.backward()

                    # For logging, use the plain (unweighted) response_mask.
                    denom_plain = response_mask.sum().clamp(min=1)

                    kl_f_mean = (kl_forward * response_mask).sum().item() / denom_plain.item()
                    kl_r_mean = (kl_reverse * response_mask).sum().item() / denom_plain.item()
                    jsd_mean = (jsd_tok * response_mask).sum().item() / denom_plain.item()
                    loss_coverage_mean = (loss_coverage * response_mask).sum().item() / denom_plain.item()

                    entropy_s_mean = (entropy_student_tok * response_mask).sum().item() / denom_plain.item()
                    entropy_t_mean = (entropy_teacher_tok * response_mask).sum().item() / denom_plain.item()

                    append_to_dict(
                        metrics,
                        {
                            "actor/distill_loss": loss.detach().item(),
                            "actor/kl_forward_mean": kl_f_mean,
                            "actor/kl_reverse_mean": kl_r_mean,
                            "actor/jsd_mean": jsd_mean,
                            "actor/jsd_alpha": jsd_alpha,
                            "actor/loss_coverage_mean": loss_coverage_mean,
                            "actor/power_alpha": float(self.power_alpha),
                            "actor/power_w_entropy": float(-(w * (w + 1e-12).log()).sum().item()),
                            "actor/power_w_max": float(w.max().item()),
                            "actor/student_entropy_mean": entropy_s_mean,
                            "actor/teacher_entropy_mean": entropy_t_mean,
                        },
                    )

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
