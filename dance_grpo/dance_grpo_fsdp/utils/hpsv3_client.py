# hps_client.py
import os
import time
from typing import Optional

import requests

# 指定使用GPU 1
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "9"


class HPSv3Client:
    def __init__(self, server_url: str = "http://localhost:5001"):
        """
        初始化HPSv3客户端

        Args:
            server_url: 服务端地址，如 "http://192.168.1.100:5001"
        """
        self.server_url = server_url.rstrip("/")

    def get_score(self, images: list[str], prompts: list[str]) -> Optional[list[float]]:
        """
        获取图像评分（最简接口）

        Args:
            images: 图片路径列表
            prompts: 对应文本提示列表

        Returns:
            评分列表（浮点数），失败返回None
        """
        # 准备请求数据
        data = {"images": images, "prompts": prompts}

        try:
            start_time = time.time()
            response = requests.post(f"{self.server_url}/reward", json=data, timeout=60)

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    elapsed = time.time() - start_time
                    print(f"Successfully processed {len(images)} images in {elapsed:.2f} seconds")
                    return result["scores"]
                else:
                    print(f"Server error: {result.get('error')}")
            else:
                print(f"HTTP error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Request failed: {e}")

        return None

    def check_server(self) -> bool:
        """检查服务是否可用"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
