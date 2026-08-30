#!/usr/bin/env bash
# 跑一个 DeepEP V1 双节点 case：预检 -> worker/leader 后台启动 -> 轮询 -> 回收日志
# V1 与 V2 驱动的差异：镜像换 deepep-v1-efa:dev；测试脚本是 tests/test_<name>.py
# （internode|low_latency）；无 JIT cache（V1 编译在 wheel 里）；分布式参数仍由
# torchrun 提供（V1 的 init_dist 读的正是 torchrun 设置的 MASTER_ADDR/PORT/
# WORLD_SIZE/RANK 四个 env，WORLD_SIZE 当节点数、RANK 当 node_rank）。
# 业务参数不传，用脚本默认值 = DeepEP 官方性能表配置
# （internode: 4096 tok/7168/top-8/256 experts；low_latency: 128/7168/top-8/288）。
# B300 说明：镜像钉的 b300-kineto-workaround 分支带 Kineto 兜底——日志若出现
# "WARNING: Kineto profiler returned 0 events"，说明分项时延是均摊值（合计值仍准），
# 报告里要注明这个口径。
# 用法: run_deepep_v1_case.sh <tag> <internode|low_latency> <port> [extra_env]
set -euo pipefail
cd "$(dirname "$0")/.."
source run/state.env
# 本脚本当前仅实现 2 节点（launch/轮询/日志回收均写死双机）；防止 NODE_COUNT=4 时静默跑出双节点数据
[ "${NODE_COUNT:-2}" = 2 ] || { echo "当前脚本仅实现 2 节点；${NODE_COUNT} 节点需按 TESTPLAN §4.2 扩展本脚本后再放开此检查" >&2; exit 3; }
TAG=$1 TEST=$2 PORT=$3 EXTRA="${4:-}"
SSH="ssh -n -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=run/ssh_known_hosts -o ServerAliveInterval=30 -o ConnectTimeout=15 -i $KEY_PATH"

[ -e "run/logs/$TAG-leader.log" ] && { echo "refuse: run/logs/$TAG-leader.log exists"; exit 3; }

# GPU 空闲预检（忽略 nccl-runner，它 sleep 不占显存）
for ip in $LEADER_PUBLIC_IP $WORKER_PUBLIC_IP; do
  used=$($SSH ubuntu@$ip "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1>1024{n++} END{print n+0}'")
  [ "$used" = "0" ] || { echo "busy GPUs on $ip"; exit 3; }
done

launch() { # role ip rank
  $SSH ubuntu@$2 "setsid nohup bash -c '
docker run --rm --gpus all --network host --ipc host --privileged \\
  --ulimit memlock=-1 --ulimit stack=67108864 \\
  --device /dev/infiniband --device /dev/gdrdrv \\
  -e NCCL_DEBUG=WARN -e NVSHMEM_DEBUG=WARN $EXTRA \\
  deepep-v1-efa:dev \\
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$3 \\
    --master_addr=$LEADER_PRIVATE_IP --master_port=$PORT \\
    /opt/deepep/tests/test_${TEST}.py
echo CASE_EXIT=\$?
' > ~/case-$TAG.log 2>&1 < /dev/null & echo launched"; }

launch worker "$WORKER_PUBLIC_IP" 1 >/dev/null
sleep 5
launch leader "$LEADER_PUBLIC_IP" 0 >/dev/null

for i in $(seq 1 60); do
  L=$($SSH ubuntu@$LEADER_PUBLIC_IP "grep -m1 CASE_EXIT ~/case-$TAG.log 2>/dev/null" || true)
  W=$($SSH ubuntu@$WORKER_PUBLIC_IP "grep -m1 CASE_EXIT ~/case-$TAG.log 2>/dev/null" || true)
  [ -n "$L" ] && [ -n "$W" ] && break
  sleep 15
done
[ -n "${L:-}" ] && [ -n "${W:-}" ] || { echo "TIMEOUT $TAG"; exit 4; }

$SSH ubuntu@$LEADER_PUBLIC_IP "cat ~/case-$TAG.log" > "run/logs/$TAG-leader.log"
$SSH ubuntu@$WORKER_PUBLIC_IP "cat ~/case-$TAG.log" > "run/logs/$TAG-worker.log"
echo "$TAG: leader=$L worker=$W"
