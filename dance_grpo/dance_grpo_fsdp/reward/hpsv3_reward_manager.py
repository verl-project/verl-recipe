# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
HPSv3-based reward manager for DanceGRPO.

Computes image quality rewards using the HPSv3 reward model (Human Preference
Score v3).  The reward model is kept on CPU by default and moved to the
accelerator only during scoring to avoid occupying GPU memory during training.
"""

from __future__ import annotations

import logging
import os
import tempfile

import torch
from PIL import Image

from verl import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class HPSv3RewardManager:
    """Reward manager that scores generated images with the HPSv3 model.

    The inferencer lives on CPU at rest and is temporarily moved to the
    accelerator device for scoring, keeping its memory footprint off the GPU
    during the rest of training.

    Args:
        checkpoint_path: Path to the HPSv3 model checkpoint.
        device: Device string for inference (e.g. ``"cuda"`` / ``"npu"``).
        reward_scale: Scalar multiplier applied to raw HPSv3 scores.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        reward_scale: float = 0.1,
    ) -> None:
        from hpsv3 import HPSv3RewardInferencer

        self.reward_scale = reward_scale
        self._infer_device = device

        # Keep model on CPU at rest; move to accelerator only when scoring.
        self.inferencer = HPSv3RewardInferencer(
            device="cpu",
            checkpoint_path=checkpoint_path,
        )
        self.inferencer.model.eval()
        for param in self.inferencer.model.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_images_to_tmp(images_pil: list[Image.Image]) -> list[str]:
        """Save PIL images to temporary PNG files and return their paths."""
        paths: list[str] = []
        for img in images_pil:
            fd, path = tempfile.mkstemp(suffix=".png")
            try:
                with os.fdopen(fd, "wb") as fh:
                    img.save(fh, format="PNG")
                paths.append(path)
            except Exception as exc:
                os.close(fd)
                logger.error("Error saving image to temp file: %s", exc)
        return paths

    @staticmethod
    def _cleanup_tmp_files(paths: list[str]) -> None:
        """Delete temporary files created by :meth:`_save_images_to_tmp`."""
        for path in paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                logger.warning("Could not delete temp file %s: %s", path, exc)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_reward(self, data) -> DataProto:
        """Score images in *data* and attach rewards to ``data.batch["rewards"]``.

        Expects the following keys in *data*:
          - ``non_tensor_batch["prompts"]``: 1-D array of prompt strings.
          - ``non_tensor_batch["all_images"]``: 1-D array of ``np.ndarray``
            images (uint8 HWC) produced by the rollout.

        The rewards tensor (shape ``[B]``) is added to ``data.batch``.

        Args:
            data: :class:`~verl.DataProto` produced by the rollout step.

        Returns:
            The same *data* object with ``data.batch["rewards"]`` populated.
        """
        prompts = data.non_tensor_batch["prompts"].tolist()
        all_images_np = data.non_tensor_batch["all_images"]
        all_images_pil = [Image.fromarray(img_np) for img_np in all_images_np]

        file_paths = self._save_images_to_tmp(all_images_pil)

        # Move reward model to accelerator for scoring.
        if self.inferencer.device == "cpu":
            self.inferencer.model.to(self._infer_device)
            self.inferencer.device = self._infer_device

        try:
            with torch.no_grad():
                raw_rewards = self.inferencer.reward(file_paths, prompts)
        finally:
            self._cleanup_tmp_files(file_paths)
            # Return model to CPU to free accelerator memory.
            self.inferencer.model.to("cpu")
            self.inferencer.device = "cpu"

        scores = [r[0].item() * self.reward_scale for r in raw_rewards]
        data.batch["rewards"] = torch.tensor(scores, dtype=torch.float32)

        return data
