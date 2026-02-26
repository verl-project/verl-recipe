pkill -9 python
ray stop --force
rm -rf /tmp/ray/*

# HCCL 相关配置
export HCCL_EXEC_TIMEOUT=7200
export HCCL_EVENT_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200
export ACL_DEVICE_SYNC_TIMEOUT=7200
export HCCL_ASYNC_ERROR_HANDLING=0
export P2P_HCCL_BUFFSIZE=30
export HCCL_BUFFSIZE=300
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050

export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1

export TASK_QUEUE_ENABLE=2 # 下发优化，图模式设置为1，非图模式设置为2
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"
export PYTHONUNBUFFERED=1

# 修改为当前需要跑的用例路径
DEFAULT_SH="examples/flash_rl/test_qwen3-30b_int8_npu.sh"
echo "Use $DEFAULT_SH"

ulimit -n 32768
mkdir -p logs

NNODES=1
NPUS_PER_NODE=16
# 修改为对应主节点IP
MASTER_ADDR="MASTER_ADDR"
# 修改为当前节点的通信网卡
SOCKET_IFNAME="SOCKET_IFNAME"
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_TIMEOUT=7200

# 获取当前节点IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')
if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # 主节点启动
  ray start --head --port 6766 --dashboard-host=$MASTER_ADDR --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # 判断 device_count 是否与 NNODES 相等
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash $DEFAULT_SH 2>&1 | tee logs/qwen3-30b_int8_npu.log
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # 子节点尝试往主节点注册ray直到成功
  while true; do
      # 尝试连接 Ray 集群
      ray start --address="$MASTER_ADDR:6766" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # 检查连接是否成功
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 600