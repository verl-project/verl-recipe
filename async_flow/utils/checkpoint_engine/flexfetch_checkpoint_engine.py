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

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Generator

import ray
import torch
import zmq
from mooncake.engine import TransferEngine

from verl.checkpoint_engine.base import CheckpointEngine, CheckpointEngineRegistry, TensorMeta
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.distributed import stateless_init_process_group
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address

log_level = os.getenv("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)

ALIGN = 4096  # TE remotebuffer的首地址要求：HCCS协议2MB对齐, RDMA协议4KB对齐


@dataclass
class MasterMetadata:
    ip: str
    zmq_send_ip: str
    zmq_send_port: int  # 接收端CE在组内广播权重bucket的port

    zmq_recv_ip: str
    zmq_recv_port: int

    dist_port: int  # 接收端CE在组内广播元数据的port


def align_up(p, a):
    """内存地址对齐函数"""
    return (p + a - 1) // a * a


@CheckpointEngineRegistry.register("flexfetch")
class FlexFetchCheckpointEngine(CheckpointEngine):
    """
    需要感知是发送端(trainer)还是接收端(replica或forward)
        1. 发送CE组, 各CE需要提供端口号、数据的内存地址
        2. 其它CE组, 只有rank=0与各发送CE拉取权重, 之后再进行组内broadcast

    Args:
        bucket_size (int): 单次传输大小
        rebuild_group (bool): 每次update是否需要重建通信组, 默认False
        is_master (bool): 是否是rank=0进程, 推理CE需要ip+port进行组内权重广播
        is_trainer (bool): 是发送CE还是推理CE
        rollout_dtype (torch.dtype): The dtype of the weights received from rollout workers. Defaults to torch.bfloat16.
    """

    def __init__(
        self,
        bucket_size: int,
        rebuild_group: bool = False,
        is_master: bool = False,
        is_trainer: bool = False,
        rollout_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        logger.debug(f"FlexFetchCheckpointEngine is_trainer={is_trainer}, is_master={is_master}")
        logger.debug(
            f"os.environ['RANK']={os.environ['RANK']}, "
            f"os.environ['LOCAL_RANK']={os.environ['LOCAL_RANK']}, "
            f"get_device_id()={get_device_id()}"
        )
        self.bucket_size = bucket_size
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.is_trainer = is_trainer
        self.is_master = is_master

        self.ckpt_max_num = 3
        self.weights_cache: dict[str, list[tuple]] = {}  # {版本号: 该版本完整权重}

        self.topic = "bucket_metadata"
        self.pyhccl = None
        self.server = None
        self.rank = None

        self.te = None
        self.te_rpc_port = None

        # 占位 socket, 用于 dist_port 预留, 防止同节点并发初始化时多个接收端 master 选到同一端口
        self.dist_port = None
        self._dist_sock = None

        self.init_te()
        self.version_id = None

    def init_te(self):
        # 发送CE, 需要获取ip与申请监听port
        self.ip = ray.util.get_node_ip_address().strip("[]")
        if self.is_trainer:
            self.te = TransferEngine()
            assert self.te.initialize(self.ip, "P2PHANDSHAKE", "ascend", "") == 0
            self.te_rpc_port = self.te.get_rpc_port()
            logger.debug(f"[Server] TE 初始化完成, RPC 端口: {self.te_rpc_port}")

            self.shared_bucket_buffers = {}
            self._start_zmq_server()
        # 其它CE, 注册接收buffer
        else:
            self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.int8, device="npu", pin_memory=True)
            if self.is_master:
                self.te = TransferEngine()
                assert self.te.initialize(self.ip, "P2PHANDSHAKE", "ascend", "") == 0

                assert self.te.register_memory(self.recv_buf.data_ptr(), self.bucket_size) == 0
                self._start_zmq_pub()
                # 持有占位 socket 直到 init_process_group 真正 bind 前一刻再释放,
                # 避免同节点并发初始化时 OS 把同一 ephemeral 端口发给多个接收端 master
                self.dist_port, self._dist_sock = get_free_port(self.ip, with_alive_sock=True)

    def _start_zmq_pub(self):
        """单个replica内, 主CE监听zmq端口"""
        self.zmq_send_port, self.listen_sock = get_free_port(self.ip)

        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.PUB)
        if is_valid_ipv6_address(self.ip):
            address = f"tcp://[{self.ip}]:{self.zmq_send_port}"
            self.zmq_socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{self.ip}:{self.zmq_send_port}"

        self.zmq_socket.bind(address)

    def _connect_zmq_sub(self, master_metadata: MasterMetadata):
        """单个replica内, 非主CE连接zmq端口"""
        assert self.rank > 0, "Master process should not connect to other processes."
        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.SUB)
        if is_valid_ipv6_address(master_metadata.zmq_recv_ip):
            address = f"tcp://[{master_metadata.zmq_recv_ip}]:{master_metadata.zmq_recv_port}"
            self.zmq_socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f"tcp://{master_metadata.zmq_recv_ip}:{master_metadata.zmq_recv_port}"

        self.zmq_socket.connect(address)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, self.topic)

    def _start_zmq_server(self):
        """发送CE, 监听zmq端口"""
        self.zmq_send_port, self.listen_sock = get_free_port(self.ip)
        context = zmq.Context()
        self.zmq_socket = context.socket(zmq.REP)
        address = f"tcp://{self.ip}:{self.zmq_send_port}"
        self.zmq_socket.bind(address)
        logger.debug(f"[Server] ZMQ SERVER 绑定 {address}")

    def prepare(self) -> MasterMetadata:
        """发送/接收CE的同步元数据"""
        return MasterMetadata(
            ip=self.ip,
            zmq_send_ip=self.ip if self.is_trainer or self.is_master else -1,
            zmq_send_port=self.zmq_send_port if self.is_trainer or self.is_master else [],
            zmq_recv_port=-1,
            zmq_recv_ip=-1,
            dist_port=self.dist_port if not self.is_trainer and self.is_master else -1,
        )

    @classmethod
    def build_topology(cls, trainer_world_size: int, rollout_world_size: int, metadata: list[dict]):
        """构建通信拓扑, 确认发送CE服务端元数据, 接收CE建联元数据

        Args:
            trainer_world_size (int): 训练woker组大小
            rollout_world_size (int): 推理woker组大小
            metadata (list[dict]): 构建拓扑所需元数据
        """

        # 配置发送组拓扑，所有训练节点均分注册权重，因此均要作为服务端作为拓扑起点
        trainer_kwargs = {
            "rank": list(range(trainer_world_size)),
            "world_size": [trainer_world_size] * trainer_world_size,
            "master_metadata": metadata[:trainer_world_size],
        }
        # 配置接收组(一个replica)拓扑， 组内主进程从服务端拉取数据，再组内广播
        rollout_metadate = []
        rollout_zmq_send_port = metadata[trainer_world_size].zmq_send_port
        rollout_zmq_send_ip = metadata[trainer_world_size].zmq_send_ip
        for rank in range(rollout_world_size):
            current_meta = metadata[trainer_world_size]
            if rank != 0:
                current_meta.zmq_recv_port = rollout_zmq_send_port
                current_meta.zmq_recv_ip = rollout_zmq_send_ip
            current_meta.zmq_recv_port_list = [data.zmq_send_port for data in metadata[:trainer_world_size]]
            current_meta.zmq_recv_ip_list = [data.zmq_send_ip for data in metadata[:trainer_world_size]]
            rollout_metadate.append(current_meta)

        rollout_kwargs = {
            "rank": list(range(rollout_world_size)),
            "world_size": [rollout_world_size] * rollout_world_size,
            "master_metadata": rollout_metadate,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        """根据拓扑创建通信组

        Args:
            rank (int): The rank of the current process.
            world_size (int): The total number of processes.
            master_metadata (MasterMetadata):
        """
        # 发送CE组通过port访问与接收CE组交互，不需要建通信组
        if self.is_trainer:
            return

        self.rank = rank
        self.world_size = world_size

        logger.debug(
            f"rank={rank}, world_size={self.world_size}, device_id={get_device_id()}, master_metadata={master_metadata}"
        )

        # 接收CE组内进程构建通信组，用于权重bucket tensor广播传输
        self.zmq_recv_port_list = master_metadata.zmq_recv_port_list
        self.zmq_recv_ip_list = master_metadata.zmq_recv_ip_list
        if self.rebuild_group or self.pyhccl is None:
            # 释放占位 socket, 使 dist_port 空出给下面的 rendezvous server 绑定;
            # 从关闭到 bind 的窗口极小, 且此进程不会再对该端口调用 get_free_port
            if self._dist_sock is not None:
                self._dist_sock.close()
                self._dist_sock = None
            self.pyhccl = stateless_init_process_group(
                master_metadata.ip,
                master_metadata.dist_port,
                rank,
                world_size=world_size,
                device=torch.npu.current_device(),
            )
        else:
            assert self.rank == rank, f"rank {rank} is not equal to self.rank {self.rank}"
            assert self.world_size == world_size, (
                f"world_size {world_size} is not equal to self.world_size {self.world_size}"
            )

        # 接收CE组非主进程CE建联，用于权重bucket元数据传输
        if self.rank > 0:
            self._connect_zmq_sub(master_metadata)
        # barrier
        signal = torch.tensor([1], dtype=torch.int8, device=torch.npu.current_device())
        self.pyhccl.all_reduce(signal)
        logger.debug(f"rank={rank}, all_reduce done")

    @torch.no_grad()
    def launch_server(self):
        """发送CE拉起权重访问服务(使用 ZMQ SERVER)"""
        if self.is_trainer and self.te is None:
            self.te = TransferEngine()
            assert self.te.initialize(self.ip, "P2PHANDSHAKE", "ascend", "") == 0
            self.te_rpc_port = self.te.get_rpc_port()
            logger.debug(f"[Server] TE 初始化完成, RPC 端口: {self.te_rpc_port}")

            self.shared_bucket_buffers = {}
            self._start_zmq_server()

        if self.is_trainer and not hasattr(self, "zmq_server_thread"):
            import threading

            self.zmq_server_thread = threading.Thread(target=self._zmq_server_loop, daemon=True)
            self.zmq_server_thread.start()
            logger.debug("[Server] ZMQ SERVER 已启动（线程运行），等待客户端连接...")

    def _zmq_server_loop(self):
        """ZMQ SERVER 循环处理客户端请求"""
        logger.debug("[Server] ZMQ SERVER 开始处理请求...")

        while True:
            try:
                # 收到客户端拉取权重请求
                connect_msg = self.zmq_socket.recv_string()
                req_version = int(connect_msg)

                # 发送指定版本权重的信息
                if req_version in self.weights_cache:
                    weight_info = self.weights_cache[req_version]["bucket_info"]
                    logger.debug(f"[Server] 返回版本 {req_version} 权重信息")
                else:
                    weight_info = {}
                    logger.error(f"[Server] 版本 {req_version} 不存在")
                response = {"peer_sid": f"{self.ip}:{self.te_rpc_port}", "weight_info": weight_info}
                self.zmq_socket.send_pyobj(response)

            except Exception as e:
                logger.error(f"[Server] 处理请求异常: {e}")
                break

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """发送CE组内各进程, 将指定版本权重均分并注册至TE引擎

        Args:
            weights : 指定版本的训练引擎权重, (layer_name, layer_weight)方式的Generator
        """
        version_id = kwargs.get("version_id", None)

        def ckpt_get_named_tensor_buckets(
            weights: Generator[tuple[str, torch.Tensor], None, None],
            version_id: int,
            bucket_bytes: int,
            rollout_dtype: torch.dtype = torch.bfloat16,
        ):
            """单个bucket构造, 实现layer权重卸载cpu以及按bucket结构打包
               单个bucket = 单个buffer(存放多个layer的权重值) + 该buffer内所有layer的信息
               单个buffer作为一条数据注册至TE引擎
               完整权重 = 多个bucket = 多条数据注册至TE引擎

            Args:
                bucket_bytes : 单个bucket的大小
                rollout_dtype : 推理引擎的权重数据类型

            Return:
                bucket_meta (dict[str, TensorMeta]): 该bucket包含layer的信息, layer名称/shape/dtype/buffer内偏移
                bucket_buffer (torch.Tensor): bucket_bytes大小的tensor, 包含多个layer对应weight
                bucket_ptr (int): buffer有效数据的起始偏移
            """
            if bucket_bytes <= 0:
                raise ValueError(f"bucket_bytes must be greater than 0, got {bucket_bytes}")

            global_bucket_idx = 0
            rank_bucket_idx = 0
            bucket_meta: dict[str, TensorMeta] = {}
            bucket_buffer, bucket_ptr, bucket_offset = None, None, 0
            target_element_size = torch.tensor([], dtype=rollout_dtype).element_size()
            for layer_name, layer_weight in weights:
                deterministic_nbytes = layer_weight.numel() * target_element_size
                # bucket放满时，处理该bucket的rank返回bucket信息; 全局bucket_idx增加
                if bucket_offset + deterministic_nbytes > bucket_bytes:
                    if global_bucket_idx % self.world_size == self.rank:
                        if bucket_meta:
                            yield bucket_meta, bucket_buffer, bucket_ptr
                        bucket_meta = {}
                        bucket_buffer, bucket_ptr = None, None
                        rank_bucket_idx += 1

                    global_bucket_idx += 1
                    bucket_offset = 0

                # 处理当前bucket的rank, 获取bucket_buffer并封装layer
                if global_bucket_idx % self.world_size == self.rank:
                    if not bucket_meta:
                        bucket_buffer, bucket_ptr = self.get_versioned_sub_buffer(version_id, rank_bucket_idx)
                        bucket_offset = 0

                    layer_weight = layer_weight.to("cpu", non_blocking=False).to(rollout_dtype)
                    # logger.debug(
                    #     f"[Server] Rank {self.rank} - Layer {layer_name} - Mean: {layer_weight.mean().item()}"
                    # )
                    bucket_buffer[bucket_offset : bucket_offset + layer_weight.nbytes].copy_(
                        layer_weight.view(-1).view(dtype=torch.uint8), non_blocking=True
                    )
                    bucket_meta[layer_name] = {
                        "shape": layer_weight.shape,
                        "dtype": layer_weight.dtype,
                        "offset": bucket_offset,
                    }
                bucket_offset += deterministic_nbytes

            # 循环结束后，yield 最后一个 Bucket（如果是本Rank）
            logger.debug(
                f"[Server] rank={self.rank} global_bucket_idx={global_bucket_idx} len(bucket_meta)={len(bucket_meta)}"
            )
            if (global_bucket_idx % self.world_size == self.rank) and bucket_meta:
                yield bucket_meta, bucket_buffer, bucket_ptr

        self.rank = int(os.environ.get("RANK"))
        self.world_size = int(os.environ.get("WORLD_SIZE"))
        logger.debug(
            f"\n[Server] rank={self.rank} 开始注册权重, "
            f"world_size={self.world_size} ckpt_versions={list(self.weights_cache.keys())}"
        )

        # 检查是否还可继续缓存，超出则 FIFO 淘汰最早版本
        if len(self.weights_cache) >= self.ckpt_max_num:
            oldest_version = next(iter(self.weights_cache.keys()))
            del self.weights_cache[oldest_version]
            logger.debug(f"缓存版本数超出最大值 {self.ckpt_max_num}，已淘汰最早版本: {oldest_version}")

        self.weights_cache[version_id] = {}
        bases = []
        capacities = []
        all_bucket_meta = []
        all_bucket_buffers = []
        for bucket_meta, bucket_buffer, bucket_ptr in ckpt_get_named_tensor_buckets(
            weights, version_id, self.bucket_size, self.rollout_dtype
        ):
            bases.append(bucket_ptr)
            capacities.append(self.bucket_size)
            all_bucket_meta.append(bucket_meta)
            all_bucket_buffers.append(bucket_buffer)
            logger.debug(f"[Server] rank={self.rank} all_bases={[hex(i) for i in bases]}")

        self.weights_cache[version_id] = {
            "bucket_data": all_bucket_buffers,
            "bucket_info": {
                "bucket_num": len(all_bucket_meta),
                "bucket_meta": all_bucket_meta,
                "bases": bases,
                "capacities": capacities,
            },
        }

        if version_id == 0 and len(capacities):
            valid_capacities = [self.bucket_size * self.ckpt_max_num for _ in capacities]
            assert self.te.batch_register_memory(bases, valid_capacities) == 0
            logger.debug(f"[Server] rank={self.rank} 权重注册完成，共 {len(bases)} 个权重")

    def get_versioned_sub_buffer(self, version_id: int, bucket_idx: int) -> torch.Tensor:
        """
        获取当前版本权重对应的tensor buffer(view切片,无数据拷贝)

        Args:
            bucket_idx: bucket索引
        Returns:
            sub_buffer: 当前版本对应的bucket_size大小的子buffer
        """
        # 版本0：初始化首地址对齐的大buffer（仅创建一次）
        if version_id == 0:
            total_buffer_size = self.ckpt_max_num * self.bucket_size
            big_buffer = torch.zeros(total_buffer_size + ALIGN, dtype=torch.uint8, pin_memory=True)
            offset = align_up(big_buffer.data_ptr(), ALIGN) - big_buffer.data_ptr()  # TE层需要首地址对齐4K或2MB
            aligned_buffer = big_buffer.narrow(0, offset, total_buffer_size)
            self.shared_bucket_buffers[bucket_idx] = aligned_buffer

        # 校验bucket是否存在
        if bucket_idx not in self.shared_bucket_buffers:
            raise KeyError(f"Bucket {bucket_idx} 未初始化, 请先在版本0中创建")

        # 获取大buffer
        big_buffer = self.shared_bucket_buffers[bucket_idx]

        # 计算当前版本的切片范围
        start = (version_id % self.ckpt_max_num) * self.bucket_size

        # 切分子buffer（narrow比view更安全，避免维度问题）
        sub_buffer = big_buffer.narrow(0, start, self.bucket_size)
        sub_buffer_ptr = big_buffer.data_ptr() + start * big_buffer.element_size()
        logger.debug(f"[Server] rank={self.rank} version_id={version_id} sub_buffer_ptr={hex(sub_buffer_ptr)}")
        return sub_buffer, sub_buffer_ptr

    @torch.no_grad()
    async def receive_weights(self, **kwargs) -> AsyncGenerator[tuple[str, torch.Tensor], None]:
        """接收CE组各进程接受指定版本权重

        Yields:
            weights : 指定版本的训练引擎权重, (layer_name, layer_weight)方式的Generator
        """
        version_id = kwargs.get("version_id", None)
        total_bytes, total_params = 0, 0
        receive_start_time = time.time()
        logger.debug(f"[Client] rank={self.rank} receive_weights start")

        # 访问所有发送端服务, 拉取权重
        for ip, zmq_port in zip(self.zmq_recv_ip_list, self.zmq_recv_port_list, strict=False):
            # 接收CE组内主进程：拉取指定版本权重，组内广播
            if self.rank == 0:
                # ========== 1. 建联 ==========
                context = zmq.Context()
                client_socket = context.socket(zmq.REQ)
                client_socket.connect(f"tcp://{ip}:{zmq_port}")
                logger.debug(f"[Client] rank={self.rank} 已连接服务端 {ip}:{zmq_port}")

                # ========== 2. zmq 发送拉取指定版本权重请求 ==========
                client_socket.send_string(str(version_id))

                # ========== 3. 解析权重信息, 断开链接 ==========
                response = client_socket.recv_pyobj()
                peer_sid = response.get("peer_sid", "")
                weight_info = response.get("weight_info", {})
                peer_src_bases = weight_info["bases"]
                peer_src_capacities = weight_info["capacities"]
                logger.debug(f"[Client] 收到权重信息: bucket_num={weight_info.get('bucket_num', 0)}")

                client_socket.close()

                # ========== 4. 该服务无权重, 组内广播停止信号 ==========
                if weight_info["bucket_num"] == 0:
                    logger.debug(f"[Client] rank={self.rank} 组内广播当前发送端 无权重数据")
                    # 发送空元数据 + 结束标志
                    empty_metadata = {"bucket_meta": [], "is_last": True}
                    self.zmq_socket.send_string(self.topic, flags=zmq.SNDMORE)
                    self.zmq_socket.send_pyobj(empty_metadata)

                # ========== 5. TE引擎读bucket tensor, 组内广播bucket tensor以及bucket meta ==========
                for i in range(weight_info["bucket_num"]):
                    logger.debug(
                        f"[Client] rank={self.rank} bases={hex(peer_src_bases[i])}, capacity={peer_src_capacities[i]}"
                    )
                    ret = self.te.transfer_sync_read(
                        peer_sid, self.recv_buf.data_ptr(), peer_src_bases[i], peer_src_capacities[i]
                    )
                    assert ret == 0, f"transfer_sync_read failed {ret}"
                    get_torch_device().synchronize()

                    metadata = {
                        "bucket_meta": weight_info["bucket_meta"][i],
                        "is_last": i == weight_info["bucket_num"] - 1,
                    }

                    total_bytes += peer_src_capacities[i]
                    total_params += len(metadata["bucket_meta"])

                    self.zmq_socket.send_string(self.topic, flags=zmq.SNDMORE)
                    self.zmq_socket.send_pyobj(metadata)
                    self.pyhccl.broadcast(self.recv_buf, src=0)
                    get_torch_device().synchronize()

                    for name, meta in metadata["bucket_meta"].items():
                        dtype, shape = meta["dtype"], meta["shape"]
                        size = dtype.itemsize * math.prod(shape)
                        tensor = self.recv_buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                        # logger.debug(f"[Client] Rank {self.rank} - Layer {name} - Mean: {tensor.mean().item()}")
                        yield name, tensor
            # 接收CE组内其它进程：接收广播数据
            else:
                while True:
                    self.zmq_socket.recv_string()
                    metadata = self.zmq_socket.recv_pyobj()
                    if len(metadata["bucket_meta"]) == 0:
                        break
                    self.pyhccl.broadcast(self.recv_buf, src=0)
                    get_torch_device().synchronize()

                    total_bytes += self.bucket_size
                    total_params += len(metadata["bucket_meta"])

                    for name, meta in metadata["bucket_meta"].items():
                        dtype, shape = meta["dtype"], meta["shape"]
                        size = dtype.itemsize * math.prod(shape)
                        tensor = self.recv_buf[meta["offset"] : meta["offset"] + size].view(dtype=dtype).view(shape)
                        # logger.debug(f"[Client] Rank {self.rank} - Layer {name} - Mean: {tensor.mean().item()}")
                        yield name, tensor
                    if metadata["is_last"]:
                        break

        time_cost = time.time() - receive_start_time
        bandwidth = total_bytes / time_cost / (1024 * 1024 * 1024)
        print(f"Rank {self.rank} receive weights done, time cost: {time_cost:.2f}s, bandwidth: {bandwidth:.2f} GB/s")

    def stop_server(self):
        """停止服务端：关闭 ZMQ、清理线程、清空缓存"""
        logger.debug(f"[Server] rank={self.rank} 开始停止服务并清理资源...")

        # ========== 1. 关闭 ZMQ 套接字 ==========
        if hasattr(self, "zmq_socket") and self.zmq_socket is not None:
            try:
                self.zmq_socket.close()
                logger.debug("[Server] 已关闭 ZMQ socket")
            except Exception as e:
                logger.error(f"[Server] 关闭 ZMQ socket 失败: {e}")
            self.zmq_socket = None

        # ========== 2. 停止 ZMQ 线程 ==========
        if self.is_trainer and hasattr(self, "zmq_server_thread"):
            # 线程无法直接 cancel，只能标记退出（你的 _zmq_server_loop 会自动退出）
            if self.zmq_server_thread.is_alive():
                logger.debug("[Server] ZMQ server 线程已停止（通过关闭 socket 自动退出）")
            self.zmq_server_thread = None

        # ========== 3. 清空权重缓存 ==========
        self.weights_cache.clear()
        logger.debug(f"[Server] rank={self.rank} 已完全停止，所有资源已清理")

    def finalize(self):
        """Destroy the HCCL process group if rebuild_group is True."""
        if self.rebuild_group:
            if self.rank >= 0:
                self.pyhccl.destroyComm(self.pyhccl.comm)
                self.pyhccl = None
            self.rank = None
            self.world_size = None
