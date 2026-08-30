#!/usr/bin/env bash
# 跑一个 NCCL case：生成内层脚本 -> docker cp 进容器 -> 后台执行 -> 轮询 -> 收日志
# 用法: run_nccl_case.sh <tag> <binary>   例: run_nccl_case.sh nccl-allreduce-r1 all_reduce_perf
# 说明: 内层命令写成脚本文件传输，避免 ssh/bash/docker 三层引号转义（曾因此翻车）
# B300 说明: -x NCCL_IB_HCA 必须转发（值 rdmap 来自镜像 ENV）——该机型 ibverbs
# 有 2 个非 EFA 设备，远端 rank 不筛选可能选错设备
set -euo pipefail
cd "$(dirname "$0")/.."
source run/state.env
TAG=$1 BIN=$2
SSH="ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=run/ssh_known_hosts -o ServerAliveInterval=30 -o ConnectTimeout=15 -i $KEY_PATH"
[ -e "run/logs/$TAG.log" ] && { echo "refuse: run/logs/$TAG.log exists"; exit 3; }

# 1) 本地生成内层脚本（变量在这里就展开完毕，后面不再有转义）
cat > /tmp/nccl-inner-$TAG.sh <<EOF
#!/bin/bash
printf "%s slots=8\n%s slots=8\n" $LEADER_PRIVATE_IP $WORKER_PRIVATE_IP > /root/hostfile
exec mpirun --allow-run-as-root -np 16 -N 8 --hostfile /root/hostfile \\
  -mca plm_rsh_args "-p 2222" \\
  -x LD_LIBRARY_PATH -x FI_PROVIDER -x NCCL_NET_PLUGIN -x NCCL_IB_HCA -x NCCL_DEBUG=WARN \\
  /opt/nccl-tests/build/$BIN -b 8 -e 8G -f 2 -g 1
EOF

# 2) 传到 leader 宿主机再 docker cp 进容器
scp -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile=run/ssh_known_hosts -i "$KEY_PATH" \
    /tmp/nccl-inner-$TAG.sh ubuntu@$LEADER_PUBLIC_IP:/tmp/ >/dev/null
$SSH -n ubuntu@$LEADER_PUBLIC_IP "docker cp /tmp/nccl-inner-$TAG.sh nccl-runner:/root/case.sh && rm /tmp/nccl-inner-$TAG.sh"

# 3) 后台执行（遵守 §0 远程长任务约定）
$SSH -n ubuntu@$LEADER_PUBLIC_IP "setsid nohup bash -c 'docker exec nccl-runner bash /root/case.sh; echo CASE_EXIT=\$?' > ~/nccl-$TAG.log 2>&1 < /dev/null & echo launched" >/dev/null

# 4) 轮询 + 收日志
for i in $(seq 1 40); do
  R=$($SSH -n ubuntu@$LEADER_PUBLIC_IP "grep -m1 CASE_EXIT ~/nccl-$TAG.log 2>/dev/null" || true)
  [ -n "$R" ] && break
  sleep 15
done
[ -n "${R:-}" ] || { echo "TIMEOUT $TAG"; exit 4; }
$SSH -n ubuntu@$LEADER_PUBLIC_IP "cat ~/nccl-$TAG.log" > "run/logs/$TAG.log"
rm -f /tmp/nccl-inner-$TAG.sh
BUSBW=$(awk '$1=="8589934592"{print $(NF-5); exit}' "run/logs/$TAG.log")
echo "$TAG: $R  busbw(8G)=${BUSBW:-N/A} GB/s"
