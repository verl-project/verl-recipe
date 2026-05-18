# Copyright 2025 Individual Contributor: furunding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FSDP-adapted version of the KL distillation loss.

Key difference between FSDP and Megatron: FSDP does not shard the vocab
dimension across tensor-parallel ranks, so the logits tensor on every rank
already contains the full vocab dimension. We can therefore run the standard
softmax / KL directly. The semantics are kept consistent with
``recipe.gkd.megatron_kl_loss.vocab_parallel_kl_divergence``:

* Use KL(P||Q), where P is the teacher (target) and Q is the student (source).
* Only compute on the top-k indices provided by the teacher (top-k distillation).
* The output is a per-token loss with shape equal to ``logits.shape[:-1]``.
"""

from __future__ import annotations

import torch


def topk_kl_divergence(
  logits: torch.Tensor,
  teacher_topk_logps: torch.Tensor,
  teacher_topk_indices: torch.Tensor,
) -> torch.Tensor:
    """Compute the per-token KL(P||Q) loss restricted to the teacher's top-k.

    Args:
        logits: Student model logits with shape ``(..., vocab_size)``.
        teacher_topk_logps: Teacher model log-probabilities on the top-k
            indices, with shape ``(..., top_k)``.
        teacher_topk_indices: Vocab indices corresponding to the teacher's
            top-k entries, with shape ``(..., top_k)`` and dtype ``long``.

    Returns:
        Per-token KL loss with shape ``logits.shape[:-1]``.
    """
    assert logits.shape[:-1] == teacher_topk_logps.shape[:-1], (
        f"logits/teacher_topk_logps leading dims mismatch: "
        f"{logits.shape} vs {teacher_topk_logps.shape}"
    )
    assert teacher_topk_logps.shape == teacher_topk_indices.shape, (
        f"teacher_topk_logps/teacher_topk_indices shape mismatch: "
        f"{teacher_topk_logps.shape} vs {teacher_topk_indices.shape}"
    )

    # Compute the student log-softmax once over the full vocab to avoid
    # repeated logsumexp evaluations.
    student_logps = torch.nn.functional.log_softmax(logits.float(), dim=-1)

    # Gather the student log-probs at the teacher's top-k indices; the
    # resulting shape equals ``teacher_topk_logps``.
    student_topk_logps = torch.gather(
        student_logps,
        dim=-1,
        index=teacher_topk_indices.long(),
    )

    teacher_topk_logps = teacher_topk_logps.to(student_topk_logps.dtype)
    teacher_topk_probs = torch.exp(teacher_topk_logps)

    # KL(P||Q) = sum_k P_k * (log P_k - log Q_k)
    per_token_kl = torch.sum(
        teacher_topk_probs * (teacher_topk_logps - student_topk_logps),
        dim=-1,
    )
    return per_token_kl
