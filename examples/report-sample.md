# P6-B300 × EFA：DeepEP V2 + NCCL + DeepEP V1 实测性能报告

测试日期：2026-09-02（UTC+8） | 执行：AI Agent 按 TESTPLAN.md（本文件为脱敏后的示例报告，实例/预约 ID 已替换为占位符）

## 1. 环境快照

| 项 | 值 |
|---|---|
| 实例 | 2 × p6-b300.48xlarge（8×B300 + 16×EFA/节点，每张 400 Gb/s），Capacity Block cr-0xxxxxxxxxxxxxxxx |
| Region/AZ | ap-northeast-2 / ap-northeast-2c |
| 实例 ID | leader i-0aaaaaaaaaaaaaaaa / worker i-0bbbbbbbbbbbbbbbb（各 17 ENI：1 ENA + 16 efa-only） |
| AMI | ami-0097a4a30ec557269（DL Base OSS Nvidia GPU AMI Ubuntu 24.04, 20260828，Description 含 P6-B300） |
| OS/内核/驱动 | Ubuntu 24.04 / 6.17.0-1020-aws / NVIDIA 595.91.07 |
| EFA 栈 | 出厂 efa.ko 3.0.0g → installer 1.50.0：efa.ko 3.3.0g / rdma-core 64.0amzn0 / libfabric 2.6.0amzn1.0 / aws-ofi-nccl 1.21.1（GinPlugin v14）；16× rdmap @ 400 Gb/s，COMP_CNTR=3 |
| 主机修正 | `PeerMappingOverride=1` 该 AMI 版本未预置（TESTPLAN §2.5 验收项抓到），经 `/etc/modprobe.d/nvidia-peermapping.conf` + `update-initramfs -u` + 重启补上 |
| 镜像 official | BUILD_REF 9c1f251118873e5dee9652cdfaf9848753cda7f1（amazon-contributing/DeepEP） |
| 镜像 pr12 | BUILD_REF b097b03799533c911a1a594fdeb82375fa8c3bd7（PR#1+PR#2） |
| 镜像 V1 | NVSHMEM_REF 6601f0bfda6f68caf6b1a65e322112ab090ccd4f + DEEPEP_V1_REF 84ccdf6c51a1095a1aed451b36c7ca9b663424ad（NGC pytorch:26.04-py3 base） |
| 运行时 | deep_ep 2.1.0 / torch 2.13.0+cu130 / NCCL 2.31.2(pip) / CUDA base **13.3.1** / TORCH_CUDA_ARCH_LIST **10.3** |
| 测试参数 | tokens 8192(prefill)/128(decode), hidden 7168, top-8, 256 experts, **12 SM**, FP8 dispatch+BF16 combine, type5(GDAKI), `NCCL_IB_HCA=rdmap` |
| GIN 后端证据 | INFO 诊断：`Loaded gin plugin Libfabric_GDAKI (v14)`、`Skipping ... type 2: NCCL_GIN_TYPE=5 requested`、`found 16 nics`、Ranks: 2 x 8 |

## 2. DeepEP V2 结果（16 rank 均值，两轮取优）

| 项 | 指标 | r1 | r2 | 报告值(取) |
|---|---|--:|--:|--:|
| official prefill | dispatch 时延 | 1048.06 µs | 1051.06 µs | **1048.06 µs** (r1) |
| | dispatch 带宽 SO/SU | 116.8/381.5 | 116.3/380.4 | **116.8 / 381.5 GB/s** |
|  | combine 时延 | 2908.81 µs | 2902.00 µs | **2908.81 µs** (r1) |
| | combine 带宽 SO/SU | 80.6/263.7 | 80.8/264.6 | **80.6 / 263.7 GB/s** |
|  | reduced combine 时延 | 3781.06 µs | 3769.75 µs | **3781.06 µs** (r1) |
| | reduced combine 带宽 SO/SU | 61.9/203.0 | 62.2/203.5 | **61.9 / 203.0 GB/s** |
| official decode | dispatch 时延 | 282.08 µs | 282.76 µs | **282.08 µs** (r1) |
| | dispatch 带宽 SO/SU | 6.2/20.9 | 6.2/20.9 | **6.2 / 20.9 GB/s** |
|  | combine 时延 | 161.58 µs | 161.74 µs | **161.58 µs** (r1) |
| | combine 带宽 SO/SU | 21.3/69.8 | 21.3/69.9 | **21.3 / 69.8 GB/s** |
|  | reduced combine 时延 | 179.16 µs | 179.57 µs | **179.16 µs** (r1) |
| | reduced combine 带宽 SO/SU | 19.2/62.9 | 19.1/62.9 | **19.2 / 62.9 GB/s** |
| pr12 prefill | dispatch 时延 | 1046.38 µs | 1046.44 µs | **1046.38 µs** (r1) |
| | dispatch 带宽 SO/SU | 116.8/382.2 | 117.0/382.1 | **116.8 / 382.2 GB/s** |
|  | combine 时延 | 2898.06 µs | 2899.44 µs | **2898.06 µs** (r1) |
| | combine 带宽 SO/SU | 80.9/264.8 | 80.8/264.6 | **80.9 / 264.8 GB/s** |
|  | reduced combine 时延 | 3785.88 µs | 3777.50 µs | **3785.88 µs** (r1) |
| | reduced combine 带宽 SO/SU | 61.9/202.7 | 62.1/203.1 | **61.9 / 202.7 GB/s** |
| pr12 decode | dispatch 时延 | 126.52 µs | 126.14 µs | **126.14 µs** (r2) |
| | dispatch 带宽 SO/SU | 14.2/46.6 | 14.2/46.6 | **14.2 / 46.6 GB/s** |
|  | combine 时延 | 161.42 µs | 161.39 µs | **161.39 µs** (r2) |
| | combine 带宽 SO/SU | 21.4/69.9 | 21.4/69.8 | **21.4 / 69.8 GB/s** |
|  | reduced combine 时延 | 178.97 µs | 179.30 µs | **179.30 µs** (r2) |
| | reduced combine 带宽 SO/SU | 19.3/63.0 | 19.2/63.0 | **19.2 / 63.0 GB/s** |
| pr12+sub1 prefill | dispatch 时延 | 1010.93 µs | 1011.21 µs | **1010.93 µs** (r1) |
| | dispatch 带宽 SO/SU | 120.9/395.5 | 121.0/395.4 | **120.9 / 395.5 GB/s** |
|  | combine 时延 | 2901.69 µs | 2906.62 µs | **2901.69 µs** (r1) |
| | combine 带宽 SO/SU | 80.9/264.4 | 80.7/263.9 | **80.9 / 264.4 GB/s** |
|  | reduced combine 时延 | 3788.62 µs | 3796.25 µs | **3788.62 µs** (r1) |
| | reduced combine 带宽 SO/SU | 61.9/202.4 | 61.7/202.1 | **61.9 / 202.4 GB/s** |
| pr12+sub1 decode | dispatch 时延 | 126.81 µs | 126.81 µs | **126.81 µs** (r1) |
| | dispatch 带宽 SO/SU | 14.3/46.4 | 14.2/46.3 | **14.3 / 46.4 GB/s** |
|  | combine 时延 | 161.26 µs | 161.32 µs | **161.26 µs** (r1) |
| | combine 带宽 SO/SU | 21.4/70.1 | 21.4/69.9 | **21.4 / 70.1 GB/s** |
|  | reduced combine 时延 | 179.23 µs | 179.54 µs | **179.23 µs** (r1) |
| | reduced combine 带宽 SO/SU | 19.2/63.1 | 19.2/62.9 | **19.2 / 63.1 GB/s** |

所有项 dispatch 轮间偏差 ≤0.3%，远低于 5% 一致性门槛。带宽为逻辑口径：SO=跨节点(EFA)，SU=节点内(NVLink)；SO 含机内流量，真实跨机流量 = SO × 1/2（2 节点）；算线速占比的分母是每 GPU **100 GB/s**（16×400 Gbps ÷ 8）。combine 为 BF16（dispatch 为 FP8），字节数约为 dispatch 的 2 倍，时延更长属预期。µs 与 p5en 版工具包直接可比（token 形状逐字相同）。

## 3. DeepEP V2 三组对比（PR 优化阶梯，两轮取优值；百分比相对 official）

**decode（128 tokens）**

| 指标 | official | pr12 | pr12+sub1 |
|---|--:|--:|--:|
| dispatch 时延 | 282.08 µs | 126.14 µs（**-55.3%**） | 126.81 µs（**-55.0%**） |
| dispatch 带宽 SO/SU (GB/s) | 6.2 / 20.9 | 14.2 / 46.6 | 14.3 / 46.4 |
| combine 时延 | 161.58 µs | 161.39 µs（-0.1%） | 161.26 µs（-0.2%） |
| combine 带宽 SO/SU (GB/s) | 21.3 / 69.8 | 21.4 / 69.8 | 21.4 / 70.1 |
| reduced combine 时延 | 179.16 µs | 179.30 µs（+0.1%） | 179.23 µs（+0.0%） |
| reduced combine 带宽 SO/SU (GB/s) | 19.2 / 62.9 | 19.2 / 63.0 | 19.2 / 63.1 |

**prefill（8192 tokens）**

| 指标 | official | pr12 | pr12+sub1 |
|---|--:|--:|--:|
| dispatch 时延 | 1048.06 µs | 1046.38 µs（-0.2%） | 1010.93 µs（-3.5%） |
| dispatch 带宽 SO/SU (GB/s) | 116.8 / 381.5 | 116.8 / 382.2 | 120.9 / 395.5 |
| combine 时延 | 2908.81 µs | 2898.06 µs（-0.4%） | 2901.69 µs（-0.2%） |
| combine 带宽 SO/SU (GB/s) | 80.6 / 263.7 | 80.9 / 264.8 | 80.9 / 264.4 |
| reduced combine 时延 | 3781.06 µs | 3785.88 µs（+0.1%） | 3788.62 µs（+0.2%） |
| reduced combine 带宽 SO/SU (GB/s) | 61.9 / 203.0 | 61.9 / 202.7 | 61.9 / 202.4 |

结论：**official 基线的 decode dispatch 在 B300 上确有回归**（282.1 µs，比同形状 p5en 的 168.6 µs 慢约 1.7 倍），PR#1+#2 打上后 126.1 µs、**省 55.3%**——反超 p5en 的 official 基线，与 p5en 打同样 PR 后的 111.6 µs 基本追平（仍慢约 13%）；`EP_NUM_SUB_PARTS=1` 在 B300 上对 decode 接近中性（+0.5%），对 prefill dispatch 有 -3.5% 的小幅收益（与 p5en 上 sub1 使 prefill 变差 +4.5% 方向相反）。combine/reduced combine 三组全部持平。与 PR 作者"优化目标是小 batch 的 dispatch 切分"的描述一致。prefill dispatch 跨机净带宽 ≈ 58.4 GB/s/GPU ≈ 58% 线速（12 SM 档；参考实测 24 SM 更高，见 §7）。

## 4. NCCL 结果（16 rank，8 GiB 消息 busbw，两轮取优）

| 原语 | r1 | r2 | 轮差 | 报告值 |
|---|--:|--:|--:|--:|
| allreduce | 870.30 | 865.86 | 0.5% | **870.30 GB/s** |
| alltoall | 179.37 | 179.52 | 0.1% | **179.52 GB/s** |
| allgather | 753.57 | 753.39 | 0.0% | **753.57 GB/s** |
| reducescatter | 754.38 | 753.37 | 0.1% | **754.38 GB/s** |

说明：allreduce busbw 870 GB/s 超过每节点 800 GB/s 网口聚合属正常——这是 busbw 口径自身的性质（混合链路归一化 + NVLS 折算高估，见 nccl-tests issue [#272](https://github.com/NVIDIA/nccl-tests/issues/272)/[#312](https://github.com/NVIDIA/nccl-tests/issues/312)），与 p5en 上 474.7 > 400 GB/s 同性质，非测量异常。

alltoall 179.5 GB/s 是本报告最硬的 EFA 证据：÷ 折算因子 (n−1)/n = algbw 191.5 GB/s（每 rank 总发送速率），16 rank / 2 节点意味着恰好一半流量跨节点，即每节点 8 × 191.5/2 ≈ **766 GB/s 跨节点线上流量，为 6.4 Tbps（800 GB/s）规格的 95.8%**，且需 16 条 rail（400 Gbps = 50 GB/s 每条）几乎全部同时承载流量。alltoall 点对点数据不可聚合，故此为硬下界而非估算。回落 TCP 的量级是个位数 GB/s，差两个数量级。此外另有 INFO 诊断的 `Selected provider is efa, fabric is efa-direct (found 16 nics)` 直接证据（正式轮 NCCL_DEBUG=WARN 无 INFO 行）。

### 4.1 带宽-消息大小曲线与小消息延迟（取优轮，out-of-place）

| 原语(轮) | 64 KiB | 1 MiB | 16 MiB | 256 MiB | 1 GiB | 8 GiB |
|---|--:|--:|--:|--:|--:|--:|
| allreduce (r1) | 2.1 | 20.3 | 105.7 | 516.6 | 770.6 | 870.3 |
| alltoall (r2) | 1.4 | 13.1 | 65.4 | 152.0 | 172.5 | 179.5 |
| allgather (r1) | 1.1 | 6.6 | 99.2 | 386.1 | 690.2 | 753.6 |
| reducescatter (r1) | 1.1 | 6.6 | 97.9 | 389.3 | 691.2 | 754.4 |

单位 GB/s（busbw）。饱和点在 1–8 GiB 档，256 MiB 仍只有峰值六成左右——比 p5en（约 256 MiB 起饱和）后移，带宽翻倍后需要更大的消息才能填满管道。完整 31 档原始数据在 `run/logs/nccl-*.log`（8B 起扫描，alltoall 小档因 16 rank 均分显示为 0 字节）。

| 原语 | 小消息延迟（size>0 各档最小耗时，两轮取小） |
|---|--:|
| allreduce | 42.9 µs |
| alltoall | 41.8 µs |
| allgather | 49.4 µs |
| reducescatter | 50.0 µs |

## 5. DeepEP V1 结果（可选项，V1_ENABLED=1；16 rank）

软件栈与 V2 完全独立：`deepep-v1-efa:dev` 镜像 = NGC pytorch:26.04 + amazon NVSHMEM
（`upstream-to-nvshmem@6601f0bf`，libfabric/EFA 传输层）+ `rauteric/DeepEP@84ccdf6c`
（b300-kineto-workaround 分支头 = remove-fence 全部内容 + Kineto 计时兜底）。
参数为官方脚本默认值（= DeepEP 官方性能表配置）。
**V1/V2 是不同 kernel、不同口径，与本报告 §2/§3 的 V2 数字不可直接对比。**

**计时口径**：本次 4 轮日志均未出现 `Kineto profiler returned 0 events`——Kineto 在此驱动栈
（595.91.07 / NGC 26.04）上工作正常，兜底未触发，分项时延为精确实测值（口径与 p5en 相同）。

### 5.1 Normal 模式（官方称呼，即高吞吐 kernel；test_internode，4096 tok/7168/top-8/256 experts，自动调优 Best 值）

| 指标（RDMA / NVL GB/s） | r1 | r2 |
|---|--:|--:|
| dispatch (FP8) | 87.23 / 284.86 | 87.25 / 284.93 |
| dispatch (BF16) | 125.85 / 410.99 | 126.05 / 411.63 |
| combine | 106.54 / 347.93 | 106.25 / 346.98 |

轮间偏差 ≤0.3%。RDMA = 跨节点 EFA 侧，NVL = 节点内 NVLink 侧。BF16 dispatch 与 combine **两个方向均破 100 GB/s**。

### 5.2 Low Latency 模式（官方称呼，即低延迟 kernel；test_low_latency，128 tok/7168/top-8/288 experts，全 rank 均值）

| 指标 | r1 | r2 |
|---|--:|--:|
| dispatch 时延 | 559.10 µs | 570.62 µs |
| dispatch 带宽 | 14.73 GB/s | 14.27 GB/s |
| combine 时延 | 518.32 µs | 514.93 µs |
| combine 带宽 | 31.29 GB/s | 31.26 GB/s |
| dispatch+combine 时延 | 1039.81 µs | 1015.21 µs |
| dispatch+combine 带宽 | 21.20 GB/s | 21.72 GB/s |

### 5.3 参考对照（不同硬件，仅供定位，非对比）

| 口径 | Normal dispatch | Normal combine | Low Latency dispatch | Low Latency combine |
|---|--:|--:|--:|--:|
| 本次 p6-b300/EFA（取优轮，RDMA） | 87.3 (FP8) / 126.1 (BF16) GB/s | 106.5 GB/s | 559 µs | 515 µs |
| p5en/EFA（同系列 kit 实测，2026-08） | 62.2 (FP8) / 72.0 (BF16) GB/s | 60.7 GB/s | 568 µs | 562 µs |
| DeepEP 官方 README（H800 + CX7 IB） | 43 GB/s | 43 GB/s | 118 µs | 195 µs |
| whn09 参考实测（p6-b300/EFA，2026-08） | 109.8 (BF16) GB/s | 101.7 GB/s | 量级相近 | 量级相近 |

Normal 模式带宽约为 p5en 的 1.7 倍（每卡 200→400 Gb/s），是官方 IB 参考的 2–2.9 倍；Low Latency 模式时延与 p5en 量级相同——瓶颈在 CPU proxy 路径（EFA 无 IBGDA），不随网卡换代明显变化，符合预期。

## 6. 异常记录

1. **`PeerMappingOverride=1` 未预置（主机层，测试前修复）**：TESTPLAN §2.5 验收发现 AMI 20260828 没有预置该 NVIDIA 驱动参数（EFA-GDA 硬前提），手工写入 `/etc/modprobe.d/nvidia-peermapping.conf` + `update-initramfs -u` + 重启后复检通过。未占用测试轮次。
2. **Kineto 兜底未触发**：镜像钉的 b300-kineto-workaround 分支在本驱动栈上不需要兜底（推测该问题与特定驱动/CUDA 组合相关），4 轮 V1 全部走 Kineto 精确计时。
3. **无 invalid 轮次**：DeepEP V2 12 轮 + NCCL 8 轮 + DeepEP V1 4 轮（另有 V2/V1 单机 smoke、2 个 INFO 诊断轮）全部 exit 0，一次通过。

## 7. 遗留问题

无阻塞项。可选后续：**24 SM prefill 加测**（B300 参考实测 dispatch 快 ~15% / reduced combine 快 ~40%，改 `--num-sms` 一个参数即可，本轮未执行）、4 节点扩展、`NCCL_TESTS_SPLIT` 纯跨节点口径加测。
