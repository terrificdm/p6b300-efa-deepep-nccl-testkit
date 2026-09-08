# P6-B300 × EFA：DeepEP V2 + NCCL 测试手册

在 2 台（或 4 台）`p6-b300.48xlarge` Capacity Block 实例上，先跑 DeepEP V2 跨节点 EP 通信测试，再跑 NCCL 集合通信测试，可选加测 DeepEP V1（legacy NVSHMEM 路线，见 §4.4），产出实测性能报告，最后清理全部资源。

本手册可交给 AI Agent 驱动执行，也可由工程师手动执行。每一步都写明"做什么 / 为什么 / 怎么判定通过"。环境搭建参考了 AWS 官方仓库 [awslabs/awsome-distributed-ai](https://github.com/awslabs/awsome-distributed-ai)（DeepEP V2 EFA 参考镜像）与 [terrificdm/p6b300-efa-deepepv2-test-runbook](https://github.com/terrificdm/p6b300-efa-deepepv2-test-runbook)（P6-B300 主机配置与 V2 实测经验）；执行本手册不需要访问它们。

**先记住 B300 和 p5en 的六个硬差异**（后面每一处都会展开，这里先立靶子，防止照搬 p5en 经验踩坑）：

| | p5en.48xlarge | p6-b300.48xlarge |
|---|---|---|
| 网卡布局 | 16 张卡，card0 = `efa`，card1..15 = `efa-only` | **17 张卡**，card0 **只能是 ENA**，card1..16 = `efa-only` |
| 每张 EFA | 200 Gb/s | **400 Gb/s**（每节点聚合 6.4 Tbps ≈ 800 GB/s） |
| ibverbs 设备 | 16 个，全是 EFA | **18 个**：16 个 `rdmap*` + 2 个非 EFA `ibp*`；必须 `NCCL_IB_HCA=rdmap` 筛选 |
| 子网 IP 消耗 | 每台 16 个 | **每台 1 个**（efa-only 不占 IPv4） |
| V2 构建 | sm90 / CUDA 13.0.2 | **sm103 / CUDA ≥ 13.3.x**（13.0.2 的 ptxas 在第一次 dispatch 才炸） |
| V1 计时 | Kineto 正常 | 部分驱动组合下 Kineto 返回 0 事件，需带兜底的分支（镜像已钉，见 §3.6；实测 595.91.07 栈上未触发） |

17（网卡）和 18（ibverbs 设备）不矛盾，是两套视角：card0 的 ENA 不是 RDMA 设备，不出现在 `ibv_devinfo -l` 里；那 2 个 `ibp*` 是 B300 机型恒有的板载非 EFA RDMA 设备，与 17 张 ENI 无对应。1+16=17 张卡，16+2=18 个 ibverbs 设备，各自都对——正因为多出这 2 个非 EFA 设备，才必须 `NCCL_IB_HCA=rdmap`。

---

## 0. 范围与约定

**适用范围**：2 或 4 台 p6-b300.48xlarge（8×B300 + 16×EFA/节点，每张 400 Gb/s），官方 Ubuntu 24.04 Deep Learning Base OSS Nvidia Driver GPU AMI（须支持 B300，见 §1.6），远程用户 `ubuntu`。其他机型/AMI 不在本手册范围内。

**角色**：
- Controller：你现在操作的机器（macOS/Linux），需要 AWS CLI v2、Bash、Python3、ssh/scp。所有命令默认在 Controller 上执行；需要在节点上执行的命令都通过 ssh 下发。
- Leader / Worker(s)：按 Instance ID 排序，第一台是 leader，其余是 worker。

**工作目录**：本目录（下称 `$WORKDIR`）。工具包自带的文件在 `scripts/` 和 `docker/`；所有运行产出放在 `$WORKDIR/run/`：

```text
docker/Dockerfile                     # DeepEP V2+nccl-tests 测试镜像（自包含构建，sm103）
docker/Dockerfile.v1                  # DeepEP V1 测试镜像（可选项，见 §3.6）
docker/Dockerfile.pr1289              # DeepEP V2 PR1+PR2+PR8+PR9 叠加镜像（可选项，见 §3.7）
scripts/generate_launch_template.py   # 生成 17 网卡（16 EFA）的启动模板 JSON
scripts/run_deepep_case.sh            # DeepEP V2 双节点 case 驱动（预检/启动/轮询/收日志）
scripts/run_deepep_v1_case.sh         # DeepEP V1 双节点 case 驱动（同骨架，V1 镜像与脚本）
scripts/run_nccl_case.sh              # NCCL case 驱动（内层命令以脚本文件传输，避免多层转义）
scripts/parse_deepep.py               # DeepEP V2 日志解析：全 rank 均值 + 轮间一致性检查
scripts/parse_deepep_v1.py            # DeepEP V1 日志解析：Normal 模式 Best 行 + Low Latency 模式全 rank 均值
run/
├── state.env        # 所有动态值（ID、IP、路径），每步更新
├── aws/             # AWS API 输出存档
├── keys/            # 本次测试生成的 keypair 私钥（不入 git）
├── host/            # 主机安装与验收日志
├── logs/            # 测试日志，命名 <tag>-<role>.log
└── results/         # 解析后的 JSON 与最终报告
```

**测试方法与环境的划分**：测试全部用官方脚本官方方法——DeepEP V2 用 DeepEP 官方自带的 `tests/elastic/test_ep.py`（torchrun 多机），NCCL 用 `NVIDIA/nccl-tests`（mpirun）；可选的 V1 加测同样只用官方自带脚本 `tests/test_internode.py` / `tests/test_low_latency.py`（torchrun 多机，默认参数即官方性能表配置，见 §4.4）。DeepEP 代码本体用 AWS 官方指定的 `amazon-contributing/DeepEP` fork（EFA 上必须，官方测试脚本在 fork 中原样保留）；镜像用本目录自带的 `docker/Dockerfile` 构建，依据 AWS 官方参考镜像整理，sm103 的构建参数（CUDA 13.3.1 + ARCH 10.3）取自 B300 上已实测跑通的组合。

**自包含说明**：Agent 执行本手册不需要读任何外部文档，所需脚本、版本号、SHA 都在本目录内。执行过程中的外部下载只有被测软件与环境本身：EFA installer（AWS 官方地址，SHA 校验；主机安装和镜像构建各下载/复用一次）、`amazon-contributing/DeepEP`（固定 commit）、`NVIDIA/nccl-tests` 与 `NVIDIA/gdrcopy`（源码编译）、torch/NCCL/NVSHMEM 的 pip wheel（固定版本号）；V1_ENABLED=1 时另加 NGC 基础镜像 `nvcr.io/nvidia/pytorch:26.04-py3` 与两个固定 commit 的 fork（`amazon-contributing/upstream-to-nvshmem`、`rauteric/DeepEP`，见附录 B）。

**三条硬规则**：
1. 花钱和开口子的操作必须先请示。共三个确认点（Gate）：**Gate A** 创建安全组/启动模板；**Gate B** 启动 P6-B300 实例和分配 EIP；**Gate C** 清理。请示时列出要创建/删除的资源和费用影响，等用户明确同意再执行。
2. 软件版本默认用附录 B 的已验证组合。这不是为了和谁对比，而是这套组合确认能跑通——CB 窗口有限，别把时间花在调试新版本兼容性上。想换版本先告知用户再动。最终用了什么版本，如实写进报告的环境快照。
3. 动态值只进 `state.env`（用 `key=value` 追加/覆盖），不写进本文档，不依赖 shell 会话记忆。每开新 shell 先 `source run/state.env`。

**远程长任务执行约定**（Controller 在 NAT 后时是硬要求，否则会静默挂死）：
- 凡是在节点上跑超过 1 分钟的任务（安装、构建、测试），必须后台化：`setsid nohup bash -c '...' > 日志 2>&1 < /dev/null &`。**`< /dev/null` 不能省**——后台进程继承 SSH channel 的 stdin 时，sshd 要等它释放才关连接，而 NAT 会把静默的长连接映射回收，ssh 客户端就永久挂死。
- ssh 一律加 `-n -o ServerAliveInterval=30 -o ServerAliveCountMax=6`。
- 进度用短连接轮询远端日志（`ssh ... 'tail -3 日志'`），不要用一条长阻塞 ssh 等结果。

**成本提醒**：CB 从预约窗口开始计费，与实例是否在跑无关，所以拖延不省钱，中断才浪费。额外费用是 EIP（每个约 $0.005/小时）和出网流量，量级可忽略。真正要紧的是把 CB 窗口内的时间用满。

---

## 1. 前置核对

做什么：确认 Controller 工具、AWS 身份、CB 状态、机型能力，把用户提供的信息落进 `state.env`。
为什么：P6-B300 按 CB 窗口计费，任何前置遗漏都会浪费窗口时间。这一步全部是只读操作，不花钱。

**1.1 需要用户提供的信息**（缺一项就问，不许猜）：

| 变量 | 说明 |
|---|---|
| `REGION` | CB 所在 Region |
| `CB_ID` | Capacity Block ID；用户只给 Region 时，列出候选让用户选 |
| `NODE_COUNT` | 2 或 4。**选 4 时，必须在 Gate A 之前先完成脚本扩展**（三个 case 驱动与 `parse_deepep.py` 当前固定 2 节点，见 §4.2/§4.3 与附录 A；驱动脚本已内置 NODE_COUNT=2 守卫，未扩展直接跑会明确报错）——扩展工作放在花钱开窗口之前做，别占用 CB 计费时间现场改脚本 |
| `ADMIN_CIDR` | 允许 SSH 的来源网段（如办公室出口 IP/32）。用户坚持 0.0.0.0/0 时要警告并让用户书面确认 |
| `V1_ENABLED` | 是否加测 DeepEP V1（0/1）。这是一条独立软件栈（见 §4.4），选 1 需额外构建第三个镜像（约 6-12 分钟/节点）并多花约 40 分钟窗口时间；选 0 则本手册所有标注"仅 V1_ENABLED=1"的步骤整体跳过 |

**1.2 Controller 工具**：`aws --version`（v2）、`python3`、`ssh -V`、`curl`。缺就停。

**1.3 AWS 身份**：

```bash
aws sts get-caller-identity | tee run/aws/caller-identity.json
```

让用户确认账号正确。

**1.4 核验 CB 与机型能力**：

```bash
aws ec2 describe-capacity-reservations --region "$REGION" \
  --capacity-reservation-ids "$CB_ID" | tee run/aws/capacity-block.json
```

判定通过：`ReservationType=capacity-block`、`InstanceType=p6-b300.48xlarge`、`State=active`、`AvailableInstanceCount >= NODE_COUNT`、`InstanceMatchCriteria=targeted`。**AZ 从这里读出**并写入 state，不许猜。

再核一次机型能力（三个数任何一个不符就停——说明机型布局和本手册假设不同，花钱之前就要发现）：

```bash
aws ec2 describe-instance-types --region "$REGION" --instance-types p6-b300.48xlarge \
  --query 'InstanceTypes[0].{Cards:NetworkInfo.MaximumNetworkCards,MaxEFA:NetworkInfo.EfaInfo.MaximumEfaInterfaces,Gpus:GpuInfo.Gpus[0].Count}' \
  | tee run/aws/instance-type.json
# 期望：Cards=17、MaxEFA=16、Gpus=8
```

**1.5 选子网**：列出该 AZ 的子网，和用户确认用哪个 VPC/子网。硬门槛：`AvailableIpAddressCount >= 8`。**不要照抄 p5en 的 `>= 16 × NODE_COUNT`**——b300 的 efa-only 接口不带 IPv4 地址，每台实例只占 1 个私网 IP，按 17×N 去要地址会造成没必要的停摆；要 8 个只是给 ENI 回收延迟留余量。

```bash
aws ec2 describe-subnets --region "$REGION" \
  --filters "Name=availability-zone,Values=$AZ" \
  --query 'Subnets[].{Id:SubnetId,Vpc:VpcId,Free:AvailableIpAddressCount,Name:Tags[?Key==`Name`]|[0].Value}'
```

**1.6 选 AMI**：

```bash
aws ec2 describe-images --region "$REGION" --owners amazon \
  --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*' \
            Name=state,Values=available Name=architecture,Values=x86_64 \
  --query 'reverse(sort_by(Images,&CreationDate))[:5].{Id:ImageId,Name:Name,Created:CreationDate,Description:Description}'
```

选这个 AMI 系列的原因：预装 NVIDIA 驱动、CUDA、Docker、nvidia-container-toolkit，并把本地 NVMe 自动组成 `/opt/dlami/nvme`，省掉大量主机准备工作。**B300 额外要求**：Description 必须明确支持 P6-B300 / Blackwell（B300 需要 595+ 驱动，AMI 太旧会在实例起来后才以 `compute_cap` 不是 10.3 或 `nvidia-smi` 不认卡的形式现形——那时 CB 已经在计费）。确认不了就停下来让用户选，不要"看起来最新就用"。记录 `AMI_ID`。

---

## 2. Host 环境部署

这一部分从零到"两台能互相跑 EFA 流量的裸机"。顺序：keypair → 安全组/启动模板（Gate A）→ 启动实例 + EIP（Gate B）→ 装 EFA 软件栈 → 主机验收。

### 2.1 生成本次测试专用 keypair

做什么：新建一个 keypair，私钥自动保存到本地。
为什么：不复用长期 key，测试结束随资源一起删除，泄露面最小。

```bash
KEY_NAME="p6b300-deepep-$(date +%Y%m%d%H%M)"
KEY_PATH="$WORKDIR/run/keys/$KEY_NAME.pem"
aws ec2 create-key-pair --region "$REGION" --key-name "$KEY_NAME" \
  --key-type ed25519 --query KeyMaterial --output text > "$KEY_PATH"
chmod 400 "$KEY_PATH"
ssh-keygen -y -f "$KEY_PATH" >/dev/null   # 能解析才算成功
```

私钥只存在于 `run/keys/`，不入 git、不上传任何地方。把 `KEY_NAME`、`KEY_PATH` 写入 state。

### 2.2 Gate A：安全组与启动模板

**先请示用户**，展示：将创建 1 个安全组（EFA 自引用全通 + 仅 `$ADMIN_CIDR` 的 TCP/22 入站）和 1 个启动模板。同意后执行。

安全组。EFA 用 OS-bypass 通信，官方要求安全组对自身全协议互通（入站+出站都要）：

```bash
SG_ID=$(aws ec2 create-security-group --region "$REGION" --vpc-id "$VPC_ID" \
  --group-name "p6b300-deepep-efa" --description "P6-B300 DeepEP EFA + SSH" \
  --query GroupId --output text)
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
  --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG_ID}]"
aws ec2 authorize-security-group-egress --region "$REGION" --group-id "$SG_ID" \
  --ip-permissions "IpProtocol=-1,UserIdGroupPairs=[{GroupId=$SG_ID}]"
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
  --protocol tcp --port 22 --cidr "$ADMIN_CIDR"
```

启动模板。P6-B300 每台 **17 张网卡**，布局固定：**card0/device0 = ENA（不写 `InterfaceType`，承载 IP 流量），card1..16 各自 device0 用 `efa-only`（纯 EFA）**。这是和 p5en 唯一一处不能靠改数字迁移的地方——b300 的主网卡不支持 EFA，**card0 写 `efa` 会被 run-instances 直接拒绝**。多 ENI 实例不能自动分配公网 IP，所以模板里不开公网，启动后再绑 EIP。根卷 500 GiB gp3（镜像构建需要空间）。用本目录自带的脚本生成，不用手写：

```bash
python3 scripts/generate_launch_template.py \
  --ami-id "$AMI_ID" --subnet-id "$SUBNET_ID" \
  --security-group-id "$SG_ID" --key-name "$KEY_NAME" \
  --capacity-block-id "$CB_ID" \
  --output run/aws/launch-template-data.json

aws ec2 create-launch-template --region "$REGION" \
  --launch-template-name "p6b300-deepep-nccl" \
  --launch-template-data "file://run/aws/launch-template-data.json" \
  | tee run/aws/gate-a-launch-template.json
# 从输出记录 LT_ID 与 LatestVersionNumber，写入 state 的 LT_ID/LT_VERSION
```

注意：CB 实例**不支持 Placement Group**，模板里不要设。创建模板后用 `--dry-run` 验证启动形状（成功的 dry-run 返回 `DryRunOperation` 错误码，这是预期）。

### 2.3 Gate B：启动实例并绑 EIP

**先请示用户**：即将启动 `NODE_COUNT` 台 p6-b300.48xlarge（CB 计费）+ 分配 `NODE_COUNT` 个 EIP。同意后：

```bash
aws ec2 run-instances --region "$REGION" --count "$NODE_COUNT" \
  --launch-template "LaunchTemplateId=$LT_ID,Version=$LT_VERSION" \
  --instance-market-options MarketType=capacity-block \
  --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CB_ID}" \
  | tee run/aws/gate-b-run-instances.json
aws ec2 wait instance-status-ok --region "$REGION" --instance-ids <全部ID>
```

启动后验收每台：`State=running`、`InstanceLifecycle=capacity-block`、**ENI 数=17、其中 `interface`（ENA）=1 且 `efa-only`=16、`efa`=0**、AZ 与 CB 一致。少于 16 个 efa-only 的实例终止重建，不要带病继续——EFA 必须在创建实例时打开，事后补不上。

绑 EIP：对每台的 **card0 主 ENI**（b300 上它是 ENA，也是唯一带 IP 的接口）分配并关联一个 EIP，保存 allocation/association ID 到 state（清理时要用）。按 Instance ID 排序确定 leader/worker，把各台的 `PUBLIC_IP`、`PRIVATE_IP` 写入 state。

SSH 连通性检查（对每台）：

```bash
ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new \
  -i "$KEY_PATH" ubuntu@$PUBLIC_IP \
  'id -un && nvidia-smi -L | wc -l && nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1'
```

期望输出 `ubuntu`、`8` 和 **`10.3`**。`compute_cap` 不是 10.3 就停——AMI 或机型不对，后面所有构建参数（sm103/CUDA 13.3.1）都建立在这个值上。

### 2.4 安装 EFA 软件栈

做什么：在每台上装固定版本的 AWS EFA installer，然后重启。
为什么：**b300 出厂 AMI 自带的 EFA 栈是 1.47.0，aws-ofi-nccl 连一个 GinPlugin 符号都没有——这一步在 b300 上是强制的，不是"版本可能偏旧"级别的建议**。DeepEP V2 在 EFA 上依赖整条链的新特性——efa 内核驱动（completion counter 支持）、`efa-nv-peermem`（GPU 内存注册给 EFA 的 GPUDirect 通路）、`gdrdrv`（GDRCopy）、libfabric（efa-direct fabric）、aws-ofi-nccl（NCCL 的 GIN 插件）。EFA installer 一次装齐并保证版本配套，这就是不逐个装包的原因。

固定版本见附录 B（EFA installer 1.50.0——B300 实测验证过的版本，SHA256 校验后才能装）。装之前顺手把出厂版本记下来（`cat /sys/module/efa/version`，预期 3.0.x），留作环境快照证据。每台执行：

```bash
# 下载（或由用户提供归档），校验 SHA256 与附录 B 一致，不一致就停
curl -fL -o aws-efa-installer-1.50.0.tar.gz \
  https://efa-installer.amazonaws.com/aws-efa-installer-1.50.0.tar.gz
sha256sum aws-efa-installer-1.50.0.tar.gz    # 必须等于附录 B 的值
tar xzf aws-efa-installer-1.50.0.tar.gz && cd aws-efa-installer
sudo ./efa_installer.sh -y --no-verify
sudo reboot
```

以上在节点上执行时按 §0 远程长任务约定后台化（安装约 2-3 分钟），日志保存为 `run/host/<role>-efa-install.txt`。重启是必须的（内核模块替换）；重启命令的 ssh 会话可能以非零断开，不代表失败，以重启后 2.5 的验收为准。

### 2.5 主机验收（硬门槛）

重启后对每台执行，全部通过才继续，输出存 `run/host/<role>-host-validation.txt`：

```bash
export PATH=/opt/amazon/efa/bin:$PATH
nvidia-smi -L | wc -l                          # =8
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1   # =10.3
cat /sys/module/efa/version                    # 3.3.0*
lsmod | grep efa_nv_peermem                    # 存在
modinfo gdrdrv | awk '$1=="version:"{print $2}'  # 2.5，且 /dev/gdrdrv 存在
ibv_devinfo -l | grep -c rdmap                 # =16（数 rdmap，别数总数！）
fi_info | grep -c "fabric: efa-direct"         # =16
cat /sys/class/infiniband/rdmap*/ports/1/rate | sort -u   # 400 Gb/sec 一种
findmnt /opt/dlami/nvme                        # 已挂载（DLAMI 自动配好）
find /usr/src -path "*/efa-*/src/efa-abi.h" -exec grep -c COMP_CNTR {} +   # >=1
grep -o 'PeerMappingOverride=1' /proc/driver/nvidia/params   # 存在（EFA-GDA 硬前提）
```

逐项含义与 b300 特有的坑：
- 8 GPU 齐、compute_cap 10.3（sm103 构建参数的前提）；
- EFA 驱动 3.3.0（completion counter 能力的载体）；
- efa_nv_peermem / gdrdrv：GPU 内存注册与直拷通路；
- **rdmap 计数必须用 `grep -c rdmap`**：b300 的 `ibv_devinfo -l` 总数是 **18**（16 个 EFA `rdmap*` + 2 个非 EFA `ibp*`），按"总数=16"去核会把健康机器判成坏的；那两个 `ibp*` 属正常存在，运行时用 `NCCL_IB_HCA=rdmap` 筛掉（已烧进 V2 镜像 ENV）；
- 16 条 efa-direct fabric：GIN type-5 的 libfabric 通路；
- **每张网卡 400 Gb/s**：这是"每 GPU 线速 100 GB/s"这个分母的证据（16×400÷8）；读到 200 说明这台不是 b300；
- `/opt/dlami/nvme`：本地盘（JIT cache 挂载点）；
- **COMP_CNTR 必须查 `/usr/src/efa-*/src/efa-abi.h`**（DKMS 源码），不要 grep `/usr/include/rdma/efa-abi.h`——那是发行版用户态头，装了 1.50 之后它照样是 0 个 COMP_CNTR，会误判。
- `PeerMappingOverride=1`：EFA-GDA 要把 EFA doorbell BAR 映射进 GPU 地址空间，NVIDIA 驱动默认拒绝，需以 `NVreg_RegistryDwords="PeerMappingOverride=1"` 加载（aws-ofi-nccl `doc/gin-getting-started.md` 列为 P5en/P6-B200/P6-B300 上 EFA-GDA 的硬前提）。**不要假设 DLAMI 预置了它**——实测 AMI 20260828（Ubuntu 24.04，Description 含 P6-B300）就没有预置；缺失时 GDAKI 初始化会显式抛异常，不会静默降级——在这里查是把问题从 CB 计费中提前到验收阶段。缺失时的修复（每台，重启后复检本清单）：

  ```bash
  echo 'options nvidia NVreg_RegistryDwords="PeerMappingOverride=1;"' \
    | sudo tee /etc/modprobe.d/nvidia-peermapping.conf
  sudo update-initramfs -u
  sudo reboot
  ```

`/opt/dlami/nvme` 如果不存在或不是预期状态：**停**，向用户展示磁盘清单，未经批准不许 mkfs/mdadm。

---

## 3. 容器镜像与测试软件

### 3.1 部署策略：测试都在容器里，DeepEP V2 与 NCCL 共用一个镜像

DeepEP V2 和 nccl-tests 装进同一个镜像：nccl-tests 编译时链接的就是 DeepEP 用的那份 pip NCCL 2.31.2，两套测试测的是完全相同的通信栈，结果可以直接互相印证；环境也只需构建和验收一次。（可选的 V1 是另一条通信栈，用独立镜像，见 §3.6。）

跨节点 mpirun 需要从 leader 容器 ssh 进 worker 容器拉起 MPI 进程，所以镜像里带了 sshd（监听 2222，host 网络下 22 被宿主机占用；密钥运行时注入，不烧进镜像）。这只是 MPI 的控制面，走 TCP；测试数据面由 NCCL 走 EFA，两者无关。

### 3.2 构建 DeepEP 镜像（本目录自带 Dockerfile）

DeepEP 官方（deepseek-ai）不提供 Dockerfile。本目录 `docker/Dockerfile` 是自包含的构建文件，依据 AWS 官方参考镜像（`awslabs/awsome-distributed-ai` 的 `micro-benchmarks/expert-parallelism/deepep-v2-benchmark/deepep.Dockerfile`，MIT-0）整理，sm103 构建参数（CUDA 13.3.1 + ARCH 10.3）已在文件里钉死，不需要访问任何第三方 repo。在**每台节点**上（镜像不推 registry，各节点本地构建）：

```bash
# 把 2.4 节校验过的同一个 EFA tarball 放到构建目录
mkdir -p ~/deepep-image && cd ~/deepep-image
# 从 Controller scp 本目录 docker/Dockerfile 过来，tarball 已在 ~/
cp ~/aws-efa-installer-1.50.0.tar.gz .

# 镜像 1：官方基线（DEEPEP_REF 默认钉在附录 B 的 fork main commit）
docker build -t deepep-v2-efa:official .

# 镜像 2：PR1+PR2 优化版（同一个 Dockerfile，只换 DEEPEP_REF；PR2 分支包含 PR1）
docker build --build-arg DEEPEP_REF=<PR12_DEEPEP_REF，见附录B> \
  -t deepep-v2-efa:pr12 .

# 镜像 3（可选）：PR1+PR2+PR8+PR9 叠加版（另一个 Dockerfile，FROM :official 增量构建，
# 须在镜像 1 之后；说明与验收见 §3.7）
docker build -f Dockerfile.pr1289 -t deepep-v2-efa:pr1289 .
```

两个镜像出自同一个 Dockerfile，除 DeepEP 代码 commit 外完全相同（同 EFA 栈、同 NCCL、同 torch），这是 §4.2 三组对比可比性的基础。首个镜像构建约 15-20 分钟（约 21 GB）；第二个命中前面所有层的 cache，只重拉代码重编译，约 5-10 分钟。多台并行构建，不要串行等。构建日志存 `run/host/<role>-image-build.txt`。镜像里装了什么、为什么（版本约束来自 AWS 官方 README）：
- **DeepEP V2（amazon-contributing/DeepEP fork，固定 commit）**：AWS 官方指定的 EFA 版本——EFA GDAKI 需要 fork 里的 unordered kernels，deepseek-ai 原版的 ordered kernels 在 EFA 上不正确。官方测试脚本 `tests/elastic/test_ep.py` 原样在 fork 内，测试方法不变
- **PyTorch 2.13 + CUDA 13.3.1 devel 基础镜像**：DeepEP V2 运行时 + JIT 编译要用的 nvcc。**13.3.1 是 sm_103 的硬下限**——13.0.2 的 ptxas 不认 `ptx.cuh` 里 `__CUDA_ARCH__ >= 1000` 分支的指令，且在第一次 dispatch 才炸，build 阶段看不出来，没有宏能绕。torch wheel 仍是 cu130（CUDA 次版本兼容，换 base 不用换 wheel）
- **NCCL 2.31.2（pip 包，非 apt libnccl2）**：≥2.31 才有 GIN 设备 API（`nccl_device.h`）；基础镜像自带的 apt NCCL 2.28 无 GIN，构建时会删掉防止串版本
- **aws-ofi-nccl（EFA installer ≥1.50 附带）**：NCCL 与 EFA 之间的桥；Dockerfile 里有硬门禁——必须导出 `ncclGinPlugin_v14`，否则构建失败（没有 v14 的镜像会静默走 CPU proxy，测的就不是 GPU 直发性能）
- **GDRCopy v2.5.2 用户态库**：GIN 初始化需要；内核模块 gdrdrv 在宿主机
- **容器内 EFA 用户态库（与宿主机同一 tarball 装出）**：用户态 libfabric/libibverbs 必须和宿主机内核模块配套
- **运行时默认值里烧了 `NCCL_IB_HCA=rdmap`**：b300 特有且必须（18 个 ibverbs 设备里筛出 16 个 EFA），见 §4.2

### 3.3 镜像验收

每台对两个镜像分别执行，存 `run/host/<role>-image-validation.txt`：

```bash
for IMG in deepep-v2-efa:official deepep-v2-efa:pr12; do
docker run --rm --entrypoint bash $IMG -lc '
cat /opt/DeepEP/BUILD_REF                 # official 对附录 B 的 DEEPEP_REF，pr12 对 PR12_DEEPEP_REF
echo "ARCH=$EP_BUILD_ARCH CUDA=$EP_BUILD_CUDA"   # 必须是 10.3 / 13.3.1
python3 -c "import deep_ep,torch; print(deep_ep.__version__, torch.__version__)"
dpkg -l | grep -c "^ii  libnccl2" || true # 必须为 0（不能混入 apt 的旧 NCCL）
nm -D --defined-only /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so | grep ncclGinPlugin_v14
nm -D --defined-only /lib/x86_64-linux-gnu/libibverbs.so.1 | grep -c comp_cntr   # =20（rdma-core 64.0 的 CE verbs）
fi_info --version | head -3'
done

# pr12 镜像另加两条：确认 PR 代码真的在里面
docker run --rm --entrypoint bash deepep-v2-efa:pr12 -lc '
grep -Rsl kMinTokensPerPart /opt/DeepEP/deep_ep/include | head -3   # PR2（在 unordered kernel 头文件里）
grep -Rsl EP_NUM_SUB_PARTS /opt/DeepEP/csrc/jit | head -3           # PR1 的 JIT env 转发'
```

pr12 的两条 grep 必须有输出。这一步防的是最隐蔽的失败：PR 没进镜像时，pr12 组照样能跑、exit 0、出数字——但测的就是官方代码，三组对比全部无效。`EP_BUILD_ARCH` 不是 10.3 说明构建参数被覆盖了（Blackwell 上跑不了 Hopper cubin，失败会出现在离原因很远的地方）。多节点同名镜像的 `BUILD_REF` 必须一致——两边跑的不是同一份代码，测试就作废了。

### 3.4 联合验收（smoke，跑正式测试前的最后一道闸）

两个递进检查，任何一个失败都不要开始正式测试：

**a. fi_pingpong**（主机层跨节点 EFA 通路，AWS 官方工具）：leader 上 `timeout 180 /opt/amazon/efa/bin/fi_pingpong -p efa` 监听，worker 上 `fi_pingpong -p efa <LEADER_PRIVATE_IP>`。两侧 exit 0。4 台时每个 worker 都对 leader 跑一遍。存 `run/host/fi-pingpong-<role>.log`。跨 AZ 配置错误会在这里以 `ibv_create_ah failed with EINVAL ... different availability zone` 现形。

**b. 单机 DeepEP 官方测试**（每台各自跑通 8 卡）：

```bash
docker run --rm --gpus all --network host --ipc host --privileged \
  --ulimit memlock=-1 --device /dev/infiniband --device /dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  deepep-v2-efa:official bash -lc \
  'python3 -u /opt/DeepEP/tests/elastic/test_ep.py --num-processes 8 --test-first-only'
```

判定：exit 0，日志有 8 个 rank 的完整 dispatch/combine 输出。存 `run/logs/single-node-<role>.log`。b300 的两个专有失败都会在这一步现形：
1. **`ptxas fatal` / `compiler.hpp:239`** → 镜像 CUDA base 低于 13.3.x，回 §3.2 查构建参数重建；
2. **`only 2 GIN GDAKI NICs have been created`** → `NCCL_IB_HCA=rdmap` 没生效。本工具包的镜像已把它烧进 ENV，正常不该出现；出现了就进容器 `printenv NCCL_IB_HCA` 核对，再查 `ibv_devinfo -l` 的设备命名。

### 3.5 NCCL 测试容器准备（每台）

nccl-tests 已经编译在镜像里（`/opt/nccl-tests/build/`），这一步只做两件事：起常驻容器、打通容器间 ssh。

每台节点起一个常驻容器并启动 sshd：

```bash
docker run -d --name nccl-runner --gpus all --network host --ipc host --privileged \
  --ulimit memlock=-1 --device /dev/infiniband --device /dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  deepep-v2-efa:official sleep infinity
docker exec nccl-runner /usr/sbin/sshd -p 2222
```

打通 ssh（Controller 上生成一把测试专用密钥，注入所有容器）：

```bash
ssh-keygen -t ed25519 -N "" -f run/keys/nccl-mpi
# 对每台节点：先把密钥 scp 到宿主机，再 docker cp 进容器
scp run/keys/nccl-mpi run/keys/nccl-mpi.pub ubuntu@$PUBLIC_IP:/tmp/
ssh ubuntu@$PUBLIC_IP '
  docker cp /tmp/nccl-mpi nccl-runner:/root/.ssh/id_ed25519
  docker exec nccl-runner chmod 600 /root/.ssh/id_ed25519
  docker cp /tmp/nccl-mpi.pub nccl-runner:/tmp/k.pub
  docker exec nccl-runner bash -c "cat /tmp/k.pub >> /root/.ssh/authorized_keys"
  rm /tmp/nccl-mpi /tmp/nccl-mpi.pub'
```

验证：leader 容器内 `ssh <WORKER_PRIVATE_IP> hostname`（ssh_config 已默认 2222 端口）能通。节点间 2222 端口的互通由安全组自引用全通规则覆盖，无需加规则。

### 3.6 DeepEP V1 镜像（仅 V1_ENABLED=1）

做什么：在每台节点上用 `docker/Dockerfile.v1` 构建第三个镜像 `deepep-v1-efa:dev`。
为什么必须是独立镜像：V1 和 V2 是两条互不相通的软件栈。V2 的跨节点流量走 NCCL GIN（aws-ofi-nccl 的 EFA-GDA）；V1 走 NVSHMEM，而 upstream NVSHMEM 没有 EFA 传输层、DeepEP 原版 internode kernel 又硬依赖 IBGDA（直接拼 mlx5 WQE，EFA 硬件无此实现）——所以 V1 在 EFA 上必须用配对的两个 fork：`amazon-contributing/upstream-to-nvshmem`（给 NVSHMEM 补 libfabric/EFA 传输层）+ `rauteric/DeepEP`（internode kernel 改用 upstream NVSHMEM API 并去掉 EFA SRD 无序传输下多余的 fence）。两半缺一不可，版本钉在附录 B。V2 镜像里虽然也带着 legacy 测试脚本，但那份 legacy 代码走 IBGDA，在 EFA 上跑不起来，不要用它测 V1。

**B300 与 p5en 版 V1 镜像的两处差异**（已写进 Dockerfile.v1，这里说明理由）：
1. DeepEP ref 钉 `b300-kineto-workaround` 分支头（附录 B 的 `V1_DEEPEP_REF`）而不是 p5en 用的 `remove-fence` 头。两者只差一个 commit：**部分驱动组合下** B300 上 Kineto profiler 会返回 0 个事件，原版 bench 代码解析不到 kernel 时间直接崩；该 commit 加了 CUDA event 计时兜底。兜底不触发时行为与 remove-fence 完全一致（2026-09-02 实测的 595.91.07 驱动 + NGC 26.04 栈上即未触发）。日志若出现 `WARNING: Kineto profiler returned 0 events` 属预期，注意口径变化（见 §4.4 取数）。
2. CUDA 架构 90 → 100：sm_100 的设备代码可在 sm_103（B300）上运行，参考镜像（whn09/ep-benchmarks-efa 的 deepep-v1-efa-b300，B300 实测验证）用的就是这条路。

宿主机无需任何改动（驱动、gdrdrv、EFA 栈都是现成的）。基础镜像 NGC pytorch:26.04 要求驱动 ≥595，b300 的 DLAMI 出厂驱动即满足，构建前顺手核一眼即可：`nvidia-smi --query-gpu=driver_version --format=csv,noheader`。

```bash
# 每台节点（tarball 复用 §2.4/§3.2 的同一个）：
scp docker/Dockerfile.v1 ubuntu@<ip>:~/deepep-image/
# 节点上（按 §0 远程长任务约定后台化，日志 run/host/<role>-v1-image-build.txt）：
cd ~/deepep-image && docker build -f Dockerfile.v1 -t deepep-v1-efa:dev .
```

p5en 实测构建约 6 分钟/节点（含 NGC base 拉取），b300 量级相同。构建内置两道硬门禁：NVSHMEM 装好后 `nvshmem-info -a` 必须含 libfabric（否则 EFA 传输层没编进去，构建失败）；两个 fork 都校验 `git rev-parse HEAD` 等于钉死的 SHA。

**验收**（每台，存 `run/host/v1-image-validation.txt`）：

```bash
docker run --rm --gpus all --entrypoint bash deepep-v1-efa:dev -lc '
cat /opt/deepep/BUILD_REF        # 两节点必须一致，且等于附录 B 的 NVSHMEM_REF/V1_DEEPEP_REF
python -c "import deep_ep; print(deep_ep.__file__)"   # import 只能在运行时验（构建期无 libcuda）
/opt/nvshmem/bin/nvshmem-info -a | grep -i -m2 libfabric
fi_info --version | head -1'
```

**单机 smoke**（leader，测 V1 镜像可用性，纯 NVLink 不走网）：

```bash
docker run --rm --gpus all --network host --ipc host --privileged   --ulimit memlock=-1 --ulimit stack=67108864   --device /dev/infiniband --device /dev/gdrdrv   deepep-v1-efa:dev python /opt/deepep/tests/test_intranode.py --num-processes 8
```

判定：exit 0。存 `run/logs/v1-intranode-smoke-leader.log`，不进对比表。

### 3.7 DeepEP V2 PR 叠加镜像 pr1289（可选）

**为什么叠加**。`amazon-contributing/DeepEP` 上有两组未合并的性能 PR，动的是不同的 kernel：PR #1+#2 优化 **dispatch**（§4.2 已详述，`:pr12` 镜像），PR #8+#9 优化 **combine**——

| PR | 改动 |
|---|---|
| #8 | scale-out 的 put 改为 remote-first 两遍扫描调度，抵消节点间的启动抖动 |
| #9 | 去掉 `num_channels_per_sm ≤ 4` 的 clamp（12 SM 下 48 → 96 channel）、GIN QP 11 → 13、forward warp 两两配对协作提交 |

两组 PR 改的文件除 README 外不相交，`git merge` 无冲突，叠加后各取所长：combine 收益全部来自 #8+#9，dispatch 收益基本来自 #1+#2。whn09/ep-benchmarks-efa 在 2×p6-b300（12 SM、type 5、默认 knob）上的实测量级供参考：叠加版 decode dispatch 277.5 → 118.1 µs（−57.4%）、decode 层总时间（dispatch + reduced combine）−37.5%、prefill 层总时间 −7.7%——是 b300 上测到的最优 decode 点。本镜像就是给测试矩阵加第四个数据点：official → pr12 → pr1289。

**镜像是什么**。`docker/Dockerfile.pr1289` 在 `:official` 之上只重做 DeepEP 一层：卸掉原有的 deep_ep，把 PR #9 head（含 #8）与 PR #2 head（含 #1）做 git merge 后重编，工具链层与 `:official` 逐字节相同。merge 的 sha 由两个 PR head 加钉死的提交身份完全决定，各节点及 whn09 得到的都是同一个 BUILD_REF（附录 B 的 `PR1289_BUILD_REF`）；构建期自带 sha 校验与四条内容断言，能建出来就说明四个 PR 都在。细节见文件头注释。

**构建**（每台节点，`:official` 建好之后；不依赖 `:pr12`；命中缓存约 5-10 分钟，日志存 `run/host/<role>-image-build-pr1289.txt`）：

```bash
cd ~/deepep-image && docker build -f Dockerfile.pr1289 -t deepep-v2-efa:pr1289 .
```

**验收**（每台，追加到 `run/host/<role>-image-validation.txt`）：先把 §3.3 那段 `for` 循环对 `deepep-v2-efa:pr1289` 跑一遍（`BUILD_REF` 须等于附录 B 的 `PR1289_BUILD_REF`，其余判据同——工具链层继承自 `:official`，理应全同），再加下面这段确认四个 PR 都在**已安装包**里（JIT 运行时读的是它，不是 `/opt/DeepEP` 源码树）：

```bash
docker run --rm --entrypoint bash deepep-v2-efa:pr1289 -lc '
cat /opt/DeepEP/BUILD_REF_PARENTS   # 两个 PR head：3c737dc...（#9 含 #8） bfbdd15...（#2 含 #1）
cat /opt/DeepEP/BUILD_BASE          # deepep-v2-efa:official
PKG=$(python3 -c "import deep_ep,os;print(os.path.dirname(deep_ep.__file__))")
strings $PKG/_C*.so | grep -c EP_NUM_SUB_PARTS                                            # PR1：>=1
grep -c kMinTokensPerPart     $PKG/include/deep_ep/impls/hybrid_dispatch_unordered.cuh    # PR2：>=1
grep -c kNumFwWarpsPerChannel $PKG/include/deep_ep/impls/hybrid_combine_unordered.cuh     # PR8/9：>=1
grep -o "kDefaultGinContextCnt *= *[0-9]*" $PKG/include/deep_ep/common/gin_resource_alloc.cuh   # PR9：= 13
'
```

四条 grep 任一为 0 或 QP 不是 13，说明镜像不是本 Dockerfile 建出来的（构建期同样四条断言不可能放过），停下查 `docker history`。两节点 `BUILD_REF` / `BUILD_REF_PARENTS` 必须逐字一致。再对 `:pr1289` 跑一次 §3.4(b) 的单机 smoke（镜像名换掉即可，日志存 `run/logs/single-node-pr1289-<role>.log`）：exit 0，且 Config 块显示 `#QPs: 13/13`（official/pr12 为 11/11）——这既是 PR #9 的运行时证据，也提前确认 merge 后的 kernel 能在 B300 上 JIT 通过，别把这个风险留到双节点正式轮。

**跑测**（驱动脚本与解析器原样可用；JIT cache 目录独立；端口接在 §4.2 的 12 轮之后，V1 的 8331 起之前）：

```bash
bash scripts/run_deepep_case.sh pr1289-prefill-r1 pr1289 pr1289_sm12 8192 8323
bash scripts/run_deepep_case.sh pr1289-decode-r1  pr1289 pr1289_sm12  128 8324
bash scripts/run_deepep_case.sh pr1289-prefill-r2 pr1289 pr1289_sm12 8192 8325
bash scripts/run_deepep_case.sh pr1289-decode-r2  pr1289 pr1289_sm12  128 8326
```

- `EP_NUM_SUB_PARTS=1` 在此镜像上**不作为优化配置测**：whn09 b300 实测它使 prefill dispatch 变差约 +100 µs、decode 中性。
- 报告里 pr1289 与 pr12 并列时分别标注 BUILD_REF：pr1289 用的是 rebase 后的 PR #2 head，与附录 B 的 `PR12_DEEPEP_REF` 底座不同（后者建在更早的 cc55cce 上，前者建在 main 的 8e7b42e 上；PR 补丁本身相同，双节点性能等价，见 Dockerfile 文件头）。§4.5 的三组对比表加一列 pr1289 即可。

参考（执行本节不需要访问）：whn09/ep-benchmarks-efa [runbook_zh.md](https://github.com/whn09/ep-benchmarks-efa/blob/main/deepep-v2-efa-official/docs/runbook_zh.md) §4.2（叠加镜像建法）、§9.7–9.10（p5en / b300 实测与可加性分析）；PR [#1](https://github.com/amazon-contributing/DeepEP/pull/1) [#2](https://github.com/amazon-contributing/DeepEP/pull/2) [#8](https://github.com/amazon-contributing/DeepEP/pull/8) [#9](https://github.com/amazon-contributing/DeepEP/pull/9)。

---

## 4. 执行测试与生成报告

### 4.1 测试执行约定

- 一次只跑一个测试（DeepEP 和 NCCL 都要独占 GPU/网卡，并发数字无效）。
- 每个正式轮次一个唯一 tag、唯一 MASTER_PORT，日志 `run/logs/<tag>-<role>.log`。已存在的 tag 日志不许覆盖；失败重试放 `run/logs/invalid/`。
- 每轮前置检查：所有节点 GPU 显存占用为 0、无残留测试容器。不干净就先清（`docker kill`），查明原因再跑。例外：`nccl-runner` 常驻容器空闲时（sleep）不占显存，属预期存在，不要误杀。
- 每个必跑项**跑 2 轮**（tag 加 `-r1`/`-r2`），跑完对比两轮：主要指标偏差在 5% 以内视为一致，取表现好的那轮作为报告值（两轮数据都留档并标明取了哪轮）。偏差超过 5% 说明环境有问题——清环境（显存归零、换端口）后重跑一轮替换，而不是从两个不可信的数里挑一个。首轮 JIT 编译耗时不计入结果。

### 4.2 DeepEP 多节点测试（官方脚本 + torchrun）

**EFA 上的 GIN 后端：type 2 与 type 5。** DeepEP V2 的跨节点流量走 NCCL GIN（GPU-Initiated Networking），GIN 在 EFA 上有两种后端：
- **type 2 = CPU proxy**：GPU 把发送请求交给 CPU 代理线程，由 CPU 驱动网卡。兼容性好，但多一跳 CPU，时延高。
- **type 5 = EFA-GDA**：GPU 直接向 EFA 网卡下发 RDMA（GDAKI 方式的 EFA 实现），全程无 CPU 参与。这才是 DeepEP V2 在 EFA 上的完整硬件能力。

`NCCL_GIN_TYPE` 不设置时 NCCL 不做类型过滤，按插件注册顺序选第一个能初始化的 GIN backend（aws-ofi-nccl 的 GDAKI 排在内建 CPU proxy 之前，proxy 只是最后的 fallback）。显式设 `NCCL_GIN_TYPE=5` 的价值在于强制指定路径——type 5 不可用时直接报错，而不是静默换一条路径，保证测的确实是 EFA-GDA。**本手册全部 DeepEP 测试用 type 5**：`NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0` 两个变量成对设置（下面的启动命令已含）。正式轮前的 INFO 诊断看到 GDAKI 插件加载且 type 2 被跳过，就是"确实跑在 type 5"的证据。

**b300 特有：`NCCL_IB_HCA=rdmap` 必须在**。该机型 ibverbs 恒有 18 个设备（16 个 EFA `rdmap*` + 2 个非 EFA `ibp*`），不筛选的话 GIN 只建得出 2 个 GDAKI NIC 直接报错。本工具包已把它烧进 V2 镜像 ENV，正常无感；但凡看到 `only 2 GIN GDAKI NICs have been created`，先查这个变量。

测试脚本就是镜像里 DeepEP 官方自带的 `/opt/DeepEP/tests/elastic/test_ep.py`。多机启动用 torchrun，**`--nproc_per_node` 必须是 1**——官方脚本自己在内部 spawn 8 个进程，torchrun 只负责把节点数和节点序号传进去。每台节点执行同一条命令，只有 `--node_rank` 不同（leader=0，worker=1..N-1）。

执行方式：用本目录 `scripts/run_deepep_case.sh` 驱动，它把一个 case 的完整生命周期封装成一条命令——GPU 空闲预检、拒绝覆盖已有 tag、worker 先 leader 后地按 §0 远程长任务约定后台启动、短连接轮询、把日志收回 `run/logs/<tag>-<role>.log`：

```bash
bash scripts/run_deepep_case.sh <tag> <official|pr12> <cache目录后缀> <tokens> <port> ["额外env"]
```

脚本前置：`run/state.env` 里有节点 IP 和 KEY_PATH（第 2 部分完成后自然满足）；cache 目录不存在会自动创建。当前实现固定 2 节点（`--nnodes=2`，脚本内置 NODE_COUNT=2 守卫，非 2 直接报错退出）。4 节点扩展范围：三个 case 驱动的 launch/轮询/日志回收循环与 hostfile、`state.env` 的 `WORKER2_*/WORKER3_*` 键（附录 A），**以及 `parse_deepep.py` 的 `parse_tag()`——它当前只读 `<tag>-leader.log` 和 `<tag>-worker.log` 两个文件，4 节点不改会静默只聚合一半 rank（唯一线索是输出里 `ranks` 为 16 而非 32）**。完整 12 轮照抄即可（交错顺序、端口不重复）：

```bash
bash scripts/run_deepep_case.sh deepep-prefill-r1     official official_sm12 8192 8311
bash scripts/run_deepep_case.sh pr12-prefill-r1       pr12     pr12_sm12     8192 8312
bash scripts/run_deepep_case.sh deepep-decode-r1      official official_sm12  128 8313
bash scripts/run_deepep_case.sh pr12-decode-r1        pr12     pr12_sm12      128 8314
bash scripts/run_deepep_case.sh pr12-sub1-prefill-r1  pr12     pr12_sm12     8192 8315 "-e EP_NUM_SUB_PARTS=1"
bash scripts/run_deepep_case.sh pr12-sub1-decode-r1   pr12     pr12_sm12      128 8316 "-e EP_NUM_SUB_PARTS=1"
bash scripts/run_deepep_case.sh deepep-prefill-r2     official official_sm12 8192 8317
bash scripts/run_deepep_case.sh pr12-prefill-r2       pr12     pr12_sm12     8192 8318
bash scripts/run_deepep_case.sh deepep-decode-r2      official official_sm12  128 8319
bash scripts/run_deepep_case.sh pr12-decode-r2        pr12     pr12_sm12      128 8320
bash scripts/run_deepep_case.sh pr12-sub1-prefill-r2  pr12     pr12_sm12     8192 8321 "-e EP_NUM_SUB_PARTS=1"
bash scripts/run_deepep_case.sh pr12-sub1-decode-r2   pr12     pr12_sm12      128 8322 "-e EP_NUM_SUB_PARTS=1"
```

每轮结束脚本打印 `CASE_EXIT=0` 和 `Ranks: 2 x 8`——两者都在才算这轮有效。可选加测：`:pr1289` 镜像（PR1+PR2+PR8+PR9 叠加）的 4 轮命令与注意事项见 §3.7，接在上面 12 轮之后执行，端口从 8323 起。

脚本内部执行的就是下面这条 docker run + torchrun（列出来供人工核对/单独执行；由 Controller 经 ssh 下发时，末尾 `>` 重定向写在 Controller 侧，日志直接落 `run/logs/`）。跑之前在任一节点 `ibv_devinfo -l` 看一眼设备名：镜像默认 `EP_NIC_NAME=rdmap101s0`，如果实际列表里没有这个名字，给 docker run 加 `-e EP_NIC_NAME=<列表里第一个 rdmap 名>`（脚本的话改 `launch()` 里的 -e 部分）。

```bash
docker run --rm --gpus all --network host --ipc host --privileged \
  --ulimit memlock=-1 --device /dev/infiniband --device /dev/gdrdrv \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v /opt/dlami/nvme/deep_ep_cache_<official|pr12>_sm12:/root/.deep_ep \
  -e NCCL_GIN_TYPE=5 -e NCCL_SYM_GIN_KERNELS_ENABLE=0 \
  -e NCCL_DEBUG=WARN \
  <矩阵"额外 env"列，如 -e EP_NUM_SUB_PARTS=1> \
  deepep-v2-efa:<official|pr12> \
  torchrun --nnodes=$NODE_COUNT --nproc_per_node=1 --node_rank=<n> \
    --master_addr=$LEADER_PRIVATE_IP --master_port=<port> \
    /opt/DeepEP/tests/elastic/test_ep.py \
    --num-tokens <tokens> --hidden 7168 --num-topk 8 --num-experts 256 \
    --num-sms 12 \
    --test-first-only \
    > run/logs/<tag>-<role>.log 2>&1
```

参数说明：
- `--num-tokens 8192 --hidden 7168 --num-topk 8 --num-experts 256` 是 DeepEP 官方 README 性能表的配置（V3 模型形状）；注意脚本默认值是 4096/top-6，必须显式传。形状与 p5en 版工具包逐字相同，所以两个机型的 µs 直接可比
- `--test-first-only`：只跑第一组模式（FP8 dispatch + BF16 combine，即官方表配置），否则会遍历 144 组组合
- `--num-sms 12`：显式指定而不用自动值，两个原因：一是与 p5en 版工具包同参数，跨机型对比才成立；二是 b300 上自动探测必然算错——脚本探测只读**一块**网卡的速率，b300 每 GPU 有 2 张 EFA，自动值会按真实带宽的一半来算。**想换 SM 数（如 24）只是改这个参数**：不用重建镜像、不用动环境，Agent 换 tag、换端口、换独立 JIT cache 目录（如 `..._sm24`）重跑即可，首轮会多几分钟 JIT 编译。b300 参考实测：prefill 在 24 SM 明显好于 12 SM（dispatch 快约 15%，reduced combine 快约 40%），所以 24 SM 加测在 b300 上比在 p5en 上更值得做。注意不同 SM 数是不同的数据点，报告里分开列，不要混
- `NCCL_GIN_TYPE=5` + `NCCL_SYM_GIN_KERNELS_ENABLE=0`：环境变量，让 NCCL GIN 走 EFA 的 GPU 直发路径，**两个必须成对设置**——这是 EFA 环境配置，不改变官方测试方法本身
- JIT cache 挂载到本地盘，避免每次起容器重新编译 kernel

必跑六项（NODE_COUNT=2 或 4 同表）：

| # | tag | 镜像 | tokens | 额外 env | 场景 |
|--:|---|---|--:|---|---|
| 1 | deepep-prefill | `:official` | 8192 | — | prefill / 高吞吐 |
| 2 | deepep-decode | `:official` | 128 | — | decode / 低延迟 |
| 3 | pr12-prefill | `:pr12` | 8192 | — | PR1+PR2 prefill |
| 4 | pr12-decode | `:pr12` | 128 | — | PR1+PR2 decode |
| 5 | pr12-sub1-prefill | `:pr12` | 8192 | `-e EP_NUM_SUB_PARTS=1` | PR1+PR2 + 单 sub-part prefill |
| 6 | pr12-sub1-decode | `:pr12` | 128 | `-e EP_NUM_SUB_PARTS=1` | PR1+PR2 + 单 sub-part decode |

**PR1+PR2 是什么、为什么测**：`amazon-contributing/DeepEP` 上两个未合并的 decode 时延优化 PR，叠加使用（PR2 分支包含 PR1）：
- **PR #2（主优化）**：修一个切分低效——decode 这类小 batch 场景下（如 128 tokens / 12 SM，每 channel 只有 3 个 token），现有代码会把一个 channel 的 token 切成超过 token 数的 part，产生空 part 和多次单 token 发送。PR2 引入 `kMinTokensPerPart`（默认 15），让 part 数不超过 `token 数 / 15`，小 batch 合并成更少的大发送。
- **PR #1（配套）**：kernel 里的 sub-part 几何参数（`EP_NUM_SUB_PARTS` 等）原本是编译期写死的宏，改代码才能调；PR1 把它们暴露成环境变量转发给 JIT，改 env 即换几何、自动重编译。PR #1 单独无效，必须和 #2 叠加。
- **`EP_NUM_SUB_PARTS=1`**：在 PR 镜像上把每 part 的 sub-part 切分关成 1（默认 2），即在 PR2 的"更少 part"基础上再关掉 sub 级切分。它给出第三个数据点，看两级切分各贡献多少。

三组两两对比就是"官方基线 → PR 优化 → PR 优化 + 调参"的性能阶梯。**六项全部 type 5、全部同参数**（tokens/hidden/topk/experts 一致，SM 统一显式 12），唯一差异是镜像和那一个 env，保证可比。预期收益集中在 decode 的 dispatch，且 **b300 上比 p5en 更明显**——b300 参考实测：官方基线的 decode dispatch 在 b300 上有回归（比同形状 p5en 慢约 1.7 倍），PR 打上后基本追平 p5en（省约 55%，p5en 上省约 34%）；`EP_NUM_SUB_PARTS=1` 在 b300 上接近中性（p5en 上还能再省几个点）。prefill 三组预期接近。

每项跑 2 轮（`-r1`/`-r2`，见 4.1）。端口从 8311 起每轮 +1，不复用。worker 先启动，隔几秒再启动 leader（torchrun rendezvous 会互相等，顺序只影响等待时间）。**JIT cache 目录按"镜像 × SM 数"分开挂载**（如 `/opt/dlami/nvme/deep_ep_cache_official_sm12`、`..._pr12_sm12`），避免编译产物串味——JIT cache key 不含实现头文件内容，official 和 pr12 共用目录会让 PR 版静默拿到官方 cubin，对比全变 no-op。建议 3-6 与 1-2 交错执行而不是按镜像分块，抵消时间漂移。

正式轮之前跑一个一次性诊断（不计入结果）：用 `:official` 镜像、tokens=128、`NCCL_DEBUG=INFO`、独立端口（如 8301），其余同上面命令，日志必须出现三样东西——16 张 EFA 网卡（`found 16 nics`）、GIN/GDAKI 插件加载（`Loaded gin plugin Libfabric_GDAKI (v14)`）、**type 2 被跳过**（`Skipping plugin ... type 2: NCCL_GIN_TYPE=5 requested`）。注意"插件加载"那行不够——两种后端都打它，插件注册不等于被选中；`Skipping ... type 2` 才是"确实跑在 type 5"的证据。没有这条证据，后面所有数字的成色都存疑。

跑起来先看输出开头的 Config 块：`Ranks: 2 x 8`（scaleout × scaleup；4 台为 `4 x 8`）确认识别到了多节点，`#SM: 12` 确认 SM 数生效。

### 4.3 NCCL 测试（容器内，leader 容器发起）

前提：3.5 的常驻容器在所有节点上运行、容器间 ssh 已通。在 leader 容器内执行：

```bash
# Controller 上执行：ssh leader 'docker exec ...'，重定向在 Controller 侧落 run/logs/
docker exec nccl-runner bash -lc '
# 4 台时 hostfile 每台一行
cat > /root/hostfile <<EOF
<LEADER_PRIVATE_IP> slots=8
<WORKER_PRIVATE_IP> slots=8
EOF
mpirun --allow-run-as-root -np <8×节点数，如 16> -N 8 \
  --hostfile /root/hostfile \
  -mca plm_rsh_args "-p 2222" \
  -x LD_LIBRARY_PATH -x FI_PROVIDER -x NCCL_NET_PLUGIN -x NCCL_IB_HCA \
  -x NCCL_NET=OFI -x NCCL_DEBUG=WARN \
  all_reduce_perf -b 8 -e 8G -f 2 -g 1
' > run/logs/nccl-<test>-r<n>.log 2>&1
```

`FI_PROVIDER=efa`、`NCCL_NET_PLUGIN=ofi`、`NCCL_IB_HCA=rdmap`（b300 必须）、`LD_LIBRARY_PATH`（含 aws-ofi-nccl 和 pip NCCL）都是镜像 ENV 自带的，`-x` 原样转发给远端 rank 即可。`NCCL_NET=OFI` 是显式赋值的硬保证——`NCCL_NET_PLUGIN` 只是优先加载外部插件、并不排他，插件初始化失败时 NCCL 会静默回落内建 socket 照常出数字；设了 `NCCL_NET` 后名字不匹配则直接报错退出，防止拿到回落 TCP 的假数据。hostfile 用各节点**私网 IP**。实际执行用 `scripts/run_nccl_case.sh <tag> <binary>` 驱动——它把内层 mpirun 写成脚本文件 docker cp 进容器再执行，**不要**手工拼 ssh+docker exec+bash 三层引号（实测会静默损坏参数：mpirun 报 executable not found 而二进制其实存在）。每轮结束打印 `CASE_EXIT=0` 和 8G 行的 busbw。完整 8 轮：

```bash
for r in r1 r2; do
  bash scripts/run_nccl_case.sh nccl-allreduce-$r      all_reduce_perf
  bash scripts/run_nccl_case.sh nccl-alltoall-$r       alltoall_perf
  bash scripts/run_nccl_case.sh nccl-allgather-$r      all_gather_perf
  bash scripts/run_nccl_case.sh nccl-reducescatter-$r  reduce_scatter_perf
done
```

必跑四项，各 2 轮（见 4.1）：

| tag | 二进制 | 说明 |
|---|---|---|
| nccl-allreduce | all_reduce_perf | 训练最常用原语，集群验收标准项 |
| nccl-alltoall | alltoall_perf | 与 EP dispatch/combine 通信模式同构，和 DeepEP 结果互证 |
| nccl-allgather | all_gather_perf | TP/SP 常用 |
| nccl-reducescatter | reduce_scatter_perf | ZeRO/FSDP 常用 |

可选加测（网络裸能力，排除 NVLink 贡献）：

```bash
-x NCCL_TESTS_SPLIT="AND 0x7"   # 每节点 8 组并行，纯跨节点流量；busbw 要乘组数
```

正式轮之前跑一个一次性诊断（不计入结果，照抄 §4.2 对 DeepEP 的做法）：任选一个原语（如 all_reduce_perf），把 `-x NCCL_DEBUG=WARN` 换成 `-x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,ENV,NET` 单独跑一轮，日志归档到 `run/logs/`，核对：

```bash
grep -iE "NET/OFI Selected provider is efa" run/logs/nccl-diag.log
# 期望：NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
# b300 的 2 个非 EFA ibp* 设备不属于 efa provider，不计入，此处仍是 16
```

注意：(a) `-i` 不能省——aws-ofi-nccl ≤1.13 是大写 `Selected Provider`（AWS 官方文档至今仍是旧串），1.14.0 起改为小写并新增 fabric 字段；(b) 只 grep `NET/OFI` 前缀会假阳性——该前缀由日志宏无条件添加（`nccl_ofi_log.h`），插件**初始化失败**的 WARN 也带它，必须核对到 `Selected provider is efa` 这一整句。

确认走 EFA 的判据（注意正式轮用 `NCCL_DEBUG=WARN`，日志里**没有** INFO 行可看）：一看量级——p6-b300 每节点 16×400 Gbps = 6.4 Tbps ≈ 800 GB/s 聚合（p5en 的 2 倍），8G busbw 达到几百 GB/s 即是 EFA，回落 TCP 只有个位数；二靠上面的一次性 INFO 诊断留下的 `Selected provider is efa` 证据。回落到 socket 的数字全部作废。

### 4.4 DeepEP V1 测试（仅 V1_ENABLED=1）

**V1/V2 是不同 kernel、不同口径，两组数字不能直接对比**，报告里分开列。V1 的价值是给出 legacy NVSHMEM 路线在 EFA 上的实测位置，可与 DeepEP 官方 README 性能表（H800 + CX7 IB）并排参考。

**测试方法**：官方脚本官方默认参数。V1 官方性能表的配置就是测试脚本的 argparse 默认值——`tests/test_internode.py`（Normal 模式，官方对高吞吐 kernel 的称呼）默认 4096 tokens/7168 hidden/top-8/256 experts = 官方 README 的 DeepSeek-V3/R1 pretraining setting；`tests/test_low_latency.py`（Low Latency 模式，官方对低延迟 kernel 的称呼）默认 128 tokens/7168/top-8/288 experts = 官方 production setting。所以**不传任何业务参数**，产出即官方口径。internode 含自动调优扫描（SM 数 × chunk 尺寸），单轮约 5 分钟；Low Latency 约 2 分钟。

**启动方式与 V2 相同是有意为之**：V1 测试的 `init_dist()` 读的正是 torchrun 会设置的 `MASTER_ADDR/MASTER_PORT/WORLD_SIZE/RANK` 四个环境变量（WORLD_SIZE 当节点数用、RANK 当 node_rank 用），所以照旧 `torchrun --nnodes=2 --nproc_per_node=1`，脚本内部自己 spawn 8 进程。"DeepEP 多机用 torchrun、NCCL 用 mpirun"的约定对 V1 同样成立。

用 `scripts/run_deepep_v1_case.sh` 驱动（与 V2 驱动同骨架：GPU 预检/拒绝覆盖 tag/worker 先启动/轮询/收日志；无 JIT cache——V1 kernel 编译在 wheel 里）：

```bash
bash scripts/run_deepep_v1_case.sh <tag> <internode|low_latency> <port> ["额外env"]
```

完整 4 轮（端口不复用，接在 V2/NCCL 使用过的端口之后）：

```bash
bash scripts/run_deepep_v1_case.sh v1-internode-r1       internode   8331
bash scripts/run_deepep_v1_case.sh v1-lowlatency-r1      low_latency 8332
bash scripts/run_deepep_v1_case.sh v1-internode-r2       internode   8333
bash scripts/run_deepep_v1_case.sh v1-lowlatency-r2      low_latency 8334
```

每轮两侧都打印 `CASE_EXIT=0` 才算有效。轮次约定同 4.1（2 轮取优、5% 一致性门槛）。

**B300 计时口径注意**：V1 镜像钉的分支带 Kineto 兜底（见 §3.6）。日志里出现 `WARNING: Kineto profiler returned 0 events` 时，Low Latency 模式的 dispatch/combine **分项**时延是合计值均摊出来的（不精确），只有 dispatch+combine **合计**时延和带宽是实测值；报告里注明这个口径，轮间一致性也按合计值判（解析器默认就是这么做的）。没出现 WARNING 则口径与 p5en 相同。

**取数**：跑 `python3 scripts/parse_deepep_v1.py run/logs`——internode 取 leader 日志自动调优的 Best 行（dispatch FP8/BF16 与 combine 的 RDMA/NVL 带宽）；Low Latency 模式取全 16 rank 的 dispatch/combine 分项与合并的时延、带宽均值；stderr 输出轮间一致性判定。RDMA 带宽即跨节点 EFA 侧，NVL 即节点内 NVLink 侧。

### 4.5 结果解析与报告

**DeepEP 取数**：跑 `python3 scripts/parse_deepep.py run/logs`——它对每个 tag 聚合全 rank 均值（dispatch/combine/reduced combine 的时延与 SO/SU 带宽），并输出 r1/r2 轮间偏差表，直接按表判定"OK/RETRY"。原理（供人工核对）：日志里行首 `*` 是 dispatch、`@` 是 combine、`+` 是 reduced combine，聚合 8×NODE_COUNT 个 rank 的均值。两轮按 4.1 对比取优，两轮原始值都进报告。

**b300 口径两条**（算带宽占比时容易踩，写进报告备注）：
- SO 带宽含机内流量（未加 `--ignore-local-traffic` 时），真实跨机流量 = SO × (N−1)/N，2 节点即 SO 的一半；
- 要算"线速占比"，分母是**每 GPU 100 GB/s**（16×400 Gbps ÷ 8 GPU）。拿 p5en 的 50 GB/s 去除会得到翻倍的占比且输出看不出任何异常。µs 与 p5en 版工具包直接可比（token 形状逐字相同）；GB/s 绝对值不可直接比（分母不同），归一化成线速占比后才可比。

**NCCL 取数**：取输出表格最大 size（8G）行的 `busbw`，两轮按 4.1 对比取优。唯一的合理性检查来自硬件规格本身：p6-b300 每节点 16×400 Gbps = 6400 Gbps ≈ 800 GB/s 聚合，大消息 busbw 如果只有个位数 GB/s，说明流量回落到了 TCP，数字作废，按第 6 部分排查。

**报告**：汇总为 `run/results/report.md`，包含：

1. 环境快照：Region/AZ/CB、实例 ID、AMI、EFA installer 版本（含出厂 1.47.0 → 1.50.0 的升级记录）、镜像 BUILD_REF、驱动/NCCL 版本（一张表）——没有这张表，性能数字就没有上下文
2. DeepEP 结果表：六个 case，**每个 case 必须给全 6 项指标**——dispatch 时延、dispatch SO/SU 带宽、combine 时延、combine SO/SU 带宽、reduced combine 时延、reduced combine SO/SU 带宽（parse_deepep.py 的输出里三个 kernel 都有，缺一项就是漏抄）——r1/r2 两轮并列，标明报告值取自哪轮
3. DeepEP 三组对比：decode 和 prefill 各一张表，行 = 上述 6 项指标，列 = official / pr12 / pr12+sub1，时延带相对 official 的百分比
4. NCCL 结果表：四个原语两轮的 8 GiB busbw、报告值取自哪轮；另加一张**带宽-消息大小曲线表**（从 31 档扫描里取 64K/1M/16M/256M/1G/8G 六个代表档的 busbw，能看出多大消息才饱和）和一张**小消息延迟表**（各原语 size>0 各档的最小耗时）。取数注意：首行常被预热污染（时延偏大数倍），alltoall 小档因按 rank 均分显示 size=0 属空操作，两者都不能当延迟下限用
5. DeepEP V1 结果表（仅 V1_ENABLED=1）：Normal 模式一张表（dispatch FP8/BF16、combine 的 RDMA+NVL 带宽，r1/r2 并列）；Low Latency 模式一张表（dispatch/combine 分项及合并的时延与带宽，若触发 Kineto 兜底须注明分项为均摊值）；注明 V1/V2 口径不可直接对比，并给 DeepEP 官方 README（H800+CX7 IB）对应数字作并排参考
6. 异常说明：所有 invalid 轮次的原因，一条不落
7. 遗留问题（如有）

报告写完先给用户看，确认数据已外带保存，再进入清理。

---

## 5. Gate C：清理

**先请示用户**，列出将删除的资源清单（实例 ID、EIP、SG、启动模板、keypair），并确认 `run/` 下的日志和报告已经存好或已拷走。同意后按顺序：

1. 终止全部实例，`aws ec2 wait instance-terminated` 等到位；存 `run/aws/gate-c-terminate.json`
2. 解绑并释放全部 EIP（用 state 里存的 association/allocation ID）；复查 `describe-addresses` 确认不存在
3. 确认实例的 EBS 卷（DeleteOnTermination）和 **17×N 个 ENI** 已随实例消失
4. 删除安全组、启动模板
5. 删除本次 keypair：`aws ec2 delete-key-pair --key-name "$KEY_NAME"`，本地私钥保留在 `run/keys/` 由用户处置
6. CB 预约本身不删（计费到期自动结束）

每步的 AWS 响应存 `run/aws/`。全部确认删干净后，本次测试才算结束。

---

## 6. 故障速查

b300 专有项在前。

| 现象 | 原因 | 处理 |
|---|---|---|
| run-instances 报接口配置不支持 | card0 写了 `efa`，或只写到 card0..15 | b300 是 17 卡、card0 只能 ENA、EFA 在 card1..16；用 generate_launch_template.py 生成 |
| 实例只有 16 个 ENI | 用了 p5en 的 16 卡布局 | 终止重建，EFA 不能事后打开 |
| `ibv_devinfo -l` 数出 18 个设备 | **正常**：16 个 EFA `rdmap*` + 2 个非 EFA `ibp*` | 验收数 `rdmap`（§2.5），运行时靠 `NCCL_IB_HCA=rdmap` 筛选 |
| `only 2 GIN GDAKI NICs have been created` | `NCCL_IB_HCA=rdmap` 没生效 | 本工具包镜像已烧进 ENV；进容器 `printenv NCCL_IB_HCA` 核对，再查设备命名 |
| `Arguments mismatch for instruction 'mov'` → `ptxas fatal` → `compiler.hpp:239` | 镜像 CUDA base 低于 13.3.x，sm_103 命中 `ptx.cuh` 的 `>= 1000` 分支。**第一次 dispatch 才炸** | 用本目录 Dockerfile 重建（默认 13.3.1）；没有宏能绕 |
| 线速占比算出约 200% | 分母用了 p5en 的 50 GB/s | b300 每 GPU 100 GB/s（§4.5） |
| grep COMP_CNTR 得 0 | grep 的是 `/usr/include/rdma/efa-abi.h`（发行版用户态头） | 查 `/usr/src/efa-*/src/efa-abi.h`（§2.5） |
| `/proc/driver/nvidia/params` 无 `PeerMappingOverride=1` | AMI 未预置（实测 20260828 就没有），EFA-GDA 硬前提缺失 | 按 §2.5 的修复命令补 modprobe 配置 + update-initramfs + 重启后复检 |
| V1 日志出现 `Kineto profiler returned 0 events` | **正常**：部分驱动组合下 B300 上 Kineto 拿不到事件，镜像自带兜底生效（不出现该行则 Kineto 正常，分项时延为精确值） | 测试继续有效；Low Latency 分项时延为均摊值，报告注明口径（§4.4） |
| 启动时报公网 IP 错误 | 多 ENI 不能自动分配公网 IP | 模板不开公网，启动后绑 EIP |
| run-instances 报 Placement Group 相关错误 | CB 不支持 PG | 模板里去掉 PG |
| `fi_info -p efa-direct` 查不到 | efa-direct 是 fabric 名不是 provider 名 | 用 `fi_info \| grep fabric` |
| 单机 test_ep 就失败 | 主机/容器某层版本不配套 | 按 2.5 和 3.3 的清单逐层核对版本 |
| torchrun 起了 64 个 rank / 跑不对 | `--nproc_per_node` 写成了 8 | 必须是 1，官方脚本自己 spawn 8 进程 |
| Config 块显示 `Ranks: 1 x 8` | 没识别到多节点 | 查 master_addr/端口连通、node_rank 是否重复 |
| exit 0 但时延翻倍 | 上一轮残留进程/显存未清 | 显存归零 + 唯一端口后重跑 |
| NCCL busbw 只有个位数 GB/s | 回落到 TCP socket | 查 `FI_PROVIDER=efa`、`NCCL_IB_HCA`、LD_LIBRARY_PATH、SG 自引用规则 |
| mpirun 连不上 worker | worker 容器 sshd 未起 / 密钥未注入 | `docker exec nccl-runner /usr/sbin/sshd -p 2222`，重做 3.5 的密钥注入 |
| 两轮偏差 >5% | 环境不稳（残留、端口复用、cache 混用） | 清环境、换端口重跑一轮替换 |
| ssh 下发长任务后永久无返回 | 后台进程继承 stdin + NAT 回收空闲连接 | 按 §0 远程长任务约定：`< /dev/null`、`-n`、ServerAlive、短连接轮询 |
| pr12 组结果和 official 完全一样 | PR 没进镜像，或两镜像共用了 JIT cache 目录（cache key 不含实现头文件） | 查 BUILD_REF、§3.3 的 PR grep、cache 目录按镜像分开 |
| V1 跨节点 NVSHMEM 初始化失败 | 用了 V2 镜像里的 legacy 代码（IBGDA 路线）| EFA 上 V1 只能用 deepep-v1-efa:dev 镜像（§3.6 的配对 fork） |
| V1 构建在 nvshmem-info 步失败 | libfabric transport 没编进去 | 查 LIBFABRIC_HOME 指向的 /opt/amazon/efa 是否装好（EFA installer 那层是否成功） |

---

## 附录 A：state.env 变量清单

以下键名是 `run_deepep_case.sh` / `run_nccl_case.sh` 直接引用的，必须逐字一致：

```text
REGION AZ CB_ID NODE_COUNT VPC_ID SUBNET_ID AMI_ID
KEY_NAME KEY_PATH ADMIN_CIDR SG_ID LT_ID LT_VERSION
LEADER_ID WORKER_ID
LEADER_PUBLIC_IP LEADER_PRIVATE_IP WORKER_PUBLIC_IP WORKER_PRIVATE_IP
LEADER_EIP_ALLOC LEADER_EIP_ASSOC WORKER_EIP_ALLOC WORKER_EIP_ASSOC
DEEPEP_REF PR12_DEEPEP_REF EFA_INSTALLER_VERSION EFA_INSTALLER_SHA256
OFFICIAL_IMAGE PR12_IMAGE
V1_ENABLED NVSHMEM_REF V1_DEEPEP_REF V1_IMAGE
```

4 节点时第 2、3 个 worker 用 WORKER2_*、WORKER3_* 命名（脚本需相应扩展，见 §4.2 的 2 节点限制说明）。

## 附录 B：固定版本

以下是已验证能跑通的软件组合。固定它们的目的不是复现谁的数据，而是别把 CB 窗口时间耗在调试版本兼容性上：

```text
EFA installer   1.50.0（B300 上实测验证；出厂 AMI 自带 1.47.0 无 GinPlugin，必须升级）
EFA tar SHA256  fa6dff8593d866866c13cb4640d9059835cd4efa427971f100ab40c97bef2841
                （实测自公开 URL 下载件；内含 efa 3.3.0 / rdma-core 64.0amzn0 /
                  libfabric 2.6.0amzn1.0 / aws-ofi-nccl 1.21.1）
DeepEP          amazon-contributing/DeepEP @ 9c1f251118873e5dee9652cdfaf9848753cda7f1
                （DEEPEP_REF，官方基线镜像；AWS 官方指定的 EFA fork，已确认 commit 存在）
DeepEP PR1+PR2  amazon-contributing/DeepEP @ b097b03799533c911a1a594fdeb82375fa8c3bd7
                （PR12_DEEPEP_REF，pr12 镜像；PR #2 分支 head，包含 PR #1，已确认存在）
                想测更新的代码：查当天 fork main 的 SHA，构建时传
                `--build-arg DEEPEP_REF=<sha>`——仍然钉 SHA，不要写 main，
                否则各节点分别构建时可能拿到不同代码导致 BUILD_REF 不一致
DeepEP PR1289   amazon-contributing/DeepEP：PR #9 head 3c737dcf0da5889ba7efd26e05b4808307cc38af（含 #8）
                merge PR #2 head bfbdd15ff448783f877cb2210cb3246c8452b05e（含 #1，rebase 后）
                -> BUILD_REF a35285f0af98856625e542df24bd17a985bc05d9（PR1289_BUILD_REF；
                   Dockerfile.pr1289 的默认值并在构建期校验。可选项，见 §3.7）
NCCL(pip)       2.31.2 / NVSHMEM(pip) 3.7.2 / torch 2.13.0+cu130
CUDA base       13.3.1-devel-ubuntu24.04 / TORCH_CUDA_ARCH_LIST=10.3（B300 / sm_103）
                sm_103 的 CUDA 下限是 13.3.x（硬约束，见 §3.2）；torch wheel 仍用 cu130，
                CUDA 次版本兼容。此组合 2026-08-29 在 2×p6-b300 实测跑通
                （terrificdm/p6b300-efa-deepepv2-test-runbook）

以下仅 V1_ENABLED=1 时使用（§3.6/§4.4；构建方法依据 whn09/ep-benchmarks-efa
的 deepep-v1-efa-b300 参考镜像，B300 实测验证）：
V1 base         nvcr.io/nvidia/pytorch:26.04-py3（CUDA 13.2，要求宿主机驱动 ≥595，
                b300 DLAMI 出厂即满足）
NVSHMEM(V1)     amazon-contributing/upstream-to-nvshmem
                @ 6601f0bfda6f68caf6b1a65e322112ab090ccd4f
                （NVSHMEM_REF，devel_enriched 分支；libfabric/EFA 传输层；
                  CUDA 架构编 100，sm_100 设备代码可在 sm_103 上运行）
DeepEP V1       rauteric/DeepEP @ 84ccdf6c51a1095a1aed451b36c7ca9b663424ad
                （V1_DEEPEP_REF，b300-kineto-workaround 分支头 = remove-fence 全部内容
                  + B300 Kineto 兜底一个 commit，见 §3.6）
GDRCopy(V1镜像) v2.5.2 / EFA installer 同上 1.50.0
```

本手册不设性能基线、不做数据对比：测出来的就是这批机器在上述软件组合下的实际性能，如实记录即可。README 里给的 B300 参考数字仅用于量级核对，不是通过标准。

## 附录 C：Agent 执行检查单

```text
[ ] 1.1-1.6 前置核对全部 PASS（含机型能力 Cards=17/MaxEFA=16/Gpus=8、AMI 支持 B300），state.env 就绪
[ ] 2.1 keypair 已生成并可解析
[ ] 2.2 Gate A 已获用户批准，SG/LT 创建完成（17 卡布局），dry-run 返回 DryRunOperation
[ ] 2.3 Gate B 已获用户批准，N 台 running，每台 17 ENI（1 ENA + 16 efa-only），EIP 绑 card0，compute_cap=10.3
[ ] 2.4-2.5 EFA 安装 + 重启 + 主机验收全部 PASS（16 rdmap、400 Gb/s、COMP_CNTR、PeerMappingOverride=1）
[ ] 3.2-3.3 两个镜像构建完成，各节点 BUILD_REF 一致，ARCH=10.3/CUDA=13.3.1，pr12 的 PR 代码 grep 有输出
[ ] 3.4 fi_pingpong、单机 test_ep 全部 PASS
[ ] 3.5 各节点 nccl-runner 常驻容器运行中，容器间 ssh(2222) 已通
[ ] 4.2 DeepEP V2 6 项 × 2 轮完成且轮间一致（≤5%），INFO 诊断证据在（含 Skipping type 2）
[ ] 4.3 NCCL 4 项 × 2 轮完成且轮间一致，INFO 诊断 `Selected provider is efa` 证据在
[ ] (V1_ENABLED=1) 3.6 V1 镜像构建/验收/单机 smoke 过，两节点 BUILD_REF 一致
[ ] (V1_ENABLED=1) 4.4 V1 4 轮完成且轮间一致（Kineto 兜底触发时已注明口径）
[ ] (可选) 3.7 pr1289 镜像构建/验收/单机 smoke 过，两节点 BUILD_REF 一致且等于 PR1289_BUILD_REF，4 轮完成且轮间一致
[ ] 4.5 report.md 生成，用户确认数据已外带
[ ] 5 Gate C 已获批准，资源清理完毕并复核（含 17×N ENI 消失）
```
