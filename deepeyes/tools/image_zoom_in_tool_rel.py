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
"""An opt-in DeepEyes crop tool accepting bounding boxes on a 0-1000 grid."""

import math

from verl.tools.image_zoom_in_tool import ImageZoomInTool


class ImageZoomInToolRelative(ImageZoomInTool):
    """Convert 0-1000 boxes to pixels in the image loaded by the base tool."""

    def _maybe_resize_bbox(self, bbox_2d: list[float], image_width: int, image_height: int) -> list[float] | None:
        if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 4:
            raise ValueError("bbox_2d must contain four numbers")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox_2d):
            raise ValueError("bbox_2d must contain four finite numbers")
        x1, y1, x2, y2 = [max(0.0, min(1000.0, float(v))) for v in bbox_2d]
        pixel_bbox = [
            x1 * image_width / 1000,
            y1 * image_height / 1000,
            x2 * image_width / 1000,
            y2 * image_height / 1000,
        ]
        return super()._maybe_resize_bbox(pixel_bbox, image_width=image_width, image_height=image_height)
