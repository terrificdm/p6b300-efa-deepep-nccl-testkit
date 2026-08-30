#!/usr/bin/env bash
# 跑一个 DeepEP 双节点 case：预检 -> worker/leader 后台启动 -> 轮询 -> 回收日志
# B300 说明：NCCL_IB_HCA=rdmap 已烧进镜像 ENV（该机型 ibverbs 有 18 个设备，
# 16 rdmap + 2 ibp，不筛选 GIN 建不齐 GDAKI NIC），这里无需再传。
# 用法: run_deepep_case.sh <tag> <image:official|pr12> <cache_dir> <tokens> <port> [extra_env]
set -euo pipefail
cd "$(dirname "$0")/.."
source run/state.env
TAG=$1 IMG=$2 CACHE=$3 TOKENS=$4 PORT=$5 EXTRA="${6:-}"
SSH="ssh -n -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=run/ssh_known_hosts -o ServerAliveInterval=30 -o ConnectTimeout=15 -i $KEY_PATH"

[ -e "run/logs/$TAG-leader.log" ] && { echo "refuse: run/logs/$TAG-leader.log exists"; exit 3; }

# GPU 空闲预检（忽略 nccl-runner，它 sleep 不占显存）；顺带确保 JIT cache 目录存在
for ip in $LEADER_PUBLIC_IP $WORKER_PUBLIC_IP; do
  used=$($SSH ubuntu@$ip "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1>1024{n++} END{print n+0}'")
  [ "$used" = "0" ] || { echo "busy GPUs on $ip"; exit 3; }
  $SSH ubuntu@$ip "sudo mkdir -p /opt/dlami/nvme/deep_ep_cache_${CACHE} && sudo chmod 777 /opt/dlami/nvme/deep_ep_cache_${CACHE}"
done

launch() { # role ip rank
  $SSH ubuntu@$2 "setsid nohup bash -c '
docker run --rm --gpus all --network host --ipc host --privileged \
  --ulimit memlock=-1 --device /dev/infiniband --device /dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v /opt/dlami/nvme/deep_ep_cache_${CACHE}:/root/.deep_ep \
  -e NCCL_GIN_TYPE=5 -e NCCL_SYM_GIN_KERNELS_ENABLE=0 \
  -e NCCL_DEBUG=WARN $EXTRA \
  deepep-v2-efa:$IMG \
  torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$3 \
    --master_addr=$LEADER_PRIVATE_IP --master_port=$PORT \
    /opt/DeepEP/tests/elastic/test_ep.py \
    --num-tokens $TOKENS --hidden 7168 --num-topk 8 --num-experts 256 \
    --num-sms 12 --test-first-only
echo CASE_EXIT=\$?
' > ~/case-$TAG.log 2>&1 < /dev/null & echo launched"; }

launch worker "$WORKER_PUBLIC_IP" 1 >/dev/null
sleep 5
launch leader "$LEADER_PUBLIC_IP" 0 >/dev/null

# 首轮含 JIT 编译（约 3 分钟），轮询上限放宽到 15 分钟
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
grep -m1 "Ranks" "run/logs/$TAG-leader.log" || true
