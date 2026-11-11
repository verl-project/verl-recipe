# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
"""Image zoom-in tool for the DeepEyes recipe.

Ported from ``verl/tools/image_zoom_in_tool.py``, which was removed from the
core library in #6302.
"""

import asyncio
import base64
import io
import logging
import os
from math import ceil, floor
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import uuid4

import requests
from PIL import Image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageZoomInTool(BaseTool):
    """Crop a region of the trajectory image and return it as a new observation."""

    # Qwen processors resize to multiples of patch_size * merge_size (32).
    DEFAULT_MIN_DIMENSION = 32

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict[str, Any]] = {}

        self.min_dimension = int(config.get("min_dimension", self.DEFAULT_MIN_DIMENSION))
        self.timeout = float(config.get("timeout", 30))
        rate_limit = max(1, int(config.get("rate_limit", 256)))
        self._semaphore = asyncio.Semaphore(rate_limit)

    # ------------------------------------------------------------------
    # bbox helpers
    # ------------------------------------------------------------------
    def _validate_bbox(self, left: float, top: float, right: float, bottom: float) -> bool:
        if not (left < right and top < bottom):
            logger.warning(f"Invalid bbox shape: left={left}, top={top}, right={right}, bottom={bottom}")
            return False

        height = bottom - top
        width = right - left
        if min(height, width) == 0:
            logger.warning(f"Bbox has zero width or height: left={left}, top={top}, right={right}, bottom={bottom}")
            return False
        if max(height, width) / min(height, width) > 100:
            logger.warning(f"Bbox aspect ratio > 100: left={left}, top={top}, right={right}, bottom={bottom}")
            return False
        return True

    def _maybe_resize_bbox(self, bbox_2d: list[float], image_width: int, image_height: int) -> Optional[list[float]]:
        """Clamp the bbox into the image and enlarge it to ``min_dimension``."""
        left, top, right, bottom = (float(value) for value in bbox_2d)

        left = max(0.0, left)
        top = max(0.0, top)
        right = min(float(image_width), right)
        bottom = min(float(image_height), bottom)

        if not self._validate_bbox(left, top, right, bottom):
            return None

        current_bbox = [left, top, right, bottom]
        height = bottom - top
        width = right - left

        if height < self.min_dimension or width < self.min_dimension:
            logger.info(f"Bbox {width}x{height} is smaller than {self.min_dimension}, attempting resize.")
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0

            min_dim = min(height, width)
            if min_dim == 0:
                return None

            ratio = self.min_dimension / min_dim
            target_width = width * ratio
            target_height = height * ratio

            if target_width > image_width:
                scale_down = image_width / target_width
                target_width = image_width
                target_height *= scale_down
            if target_height > image_height:
                scale_down = image_height / target_height
                target_height = image_height
                target_width *= scale_down

            new_left = max(0.0, center_x - target_width / 2.0)
            new_top = max(0.0, center_y - target_height / 2.0)
            if new_left + target_width > image_width:
                new_left = image_width - target_width
            if new_top + target_height > image_height:
                new_top = image_height - target_height

            current_bbox = [
                floor(new_left),
                floor(new_top),
                ceil(new_left + target_width),
                ceil(new_top + target_height),
            ]

        final_left, final_top, final_right, final_bottom = current_bbox
        if not self._validate_bbox(final_left, final_top, final_right, final_bottom):
            logger.warning(f"Final bbox is invalid after processing: {current_bbox}")
            return None

        final_height = floor(final_bottom) - floor(final_top)
        final_width = floor(final_right) - floor(final_left)
        if final_height < self.min_dimension or final_width < self.min_dimension:
            logger.warning(
                f"Final bbox size ({final_width}x{final_height}) is still smaller than the minimum "
                f"({self.min_dimension}). Original bbox: {bbox_2d}, image size: {image_width}x{image_height}"
            )
            return None

        return current_bbox

    # ------------------------------------------------------------------
    # BaseTool API
    # ------------------------------------------------------------------
    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a trajectory-scoped instance that stores the source image.

        Accepts a PIL image, raw bytes, a path, an ``http(s)://`` URL, a
        ``file://`` URI or a ``data:image/...;base64,`` URI.
        """
        if instance_id is None:
            instance_id = str(uuid4())

        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        self._instance_dict[instance_id] = {"image": _load_image(image, timeout=self.timeout), "reward": 0.0}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        bbox_2d = parameters.get("bbox_2d")
        label = parameters.get("label", "")

        if not isinstance(bbox_2d, list | tuple) or len(bbox_2d) != 4:
            return (
                ToolResponse(text="Error: bbox_2d parameter is missing or not a list of 4 numbers."),
                -0.05,
                {"success": False},
            )

        instance_data = self._instance_dict[instance_id]
        image = instance_data["image"]
        image_width, image_height = image.size

        try:
            async with self._semaphore:
                resized_bbox = self._maybe_resize_bbox(bbox_2d, image_width=image_width, image_height=image_height)
                if resized_bbox is None:
                    error_msg = (
                        f"Error: The specified bounding box {bbox_2d} is invalid or results in a crop smaller "
                        f"than the minimum size of {self.min_dimension}x{self.min_dimension}."
                    )
                    logger.warning(f"Tool execution failed: {error_msg}")
                    return ToolResponse(text=error_msg), -0.05, {"success": False}
                cropped_image = image.crop(resized_bbox)
                logger.info(f"Cropped image size: {cropped_image.size}")
        except Exception as e:
            logger.error(f"Error processing image zoom-in: {e}")
            return ToolResponse(text=f"Error processing image zoom-in: {e}"), -0.05, {"success": False}

        response_text = f"Zoomed in on the image to the region {bbox_2d}."
        if label:
            response_text = f"Zoomed in on the image to the region {bbox_2d} with label {label}."

        return ToolResponse(image=[cropped_image], text=response_text), 0.0, {"success": True}

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instance_dict.pop(instance_id, None)


def _load_image(image: Any, timeout: float = 30.0) -> Image.Image:
    """Resolve an image reference into an RGB ``PIL.Image``."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, dict):
        for key in ("image", "image_url"):
            if image.get(key) is not None:
                return _load_image(image[key], timeout=timeout)
        raise TypeError(f"Unsupported image dict: {sorted(image.keys())}")

    if isinstance(image, bytes | bytearray):
        return Image.open(io.BytesIO(image)).convert("RGB")

    if isinstance(image, str):
        if image.startswith("data:image/"):
            encoded = image.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        if image.startswith(("http://", "https://")):
            response = requests.get(image, timeout=timeout)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert("RGB")
        if image.startswith("file://"):
            image = urlparse(image).path
        return Image.open(image).convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")
