# p6b300-efa-deepep-nccl-testkit

在 AWS `p6-b300.48xlarge`（8×B300 + 16×EFA/节点，每张 400 Gb/s，Capacity Block）上测试 **DeepEP V2** 跨节点 EP 通信与 **NCCL** 集合通信实际性能的自包含工具包，并提供 **DeepEP V1**（legacy NVSHMEM 路线）作为可选加测项。从拉起实例、装 EFA 栈、构建镜像，到跑官方测试脚本、解析出报告、清理资源，全流程按 TESTPLAN.md 执行即可，每一步都给出具体命令与通过判据，适合交给 AI Agent 驱动，也适合工程师手动执行。

本工具包是 [p5en-efa-deepep-nccl-testkit](https://github.com/terrificdm/p5en-efa-deepep-nccl-testkit) 的 P6-B300 版：目录结构、测试矩阵、执行方法与 p5en 版一致（两个机型的 µs 结果直接可比），机型差异全部落实在文档与脚本内部。

## 从这里开始

**[TESTPLAN.md](TESTPLAN.md)** —— 完整执行手册（中文）。结构：前置核对 → Host 部署（Gate A/B）→ 镜像与验收 → 执行测试与报告 → 清理（Gate C）→ 故障速查。

## 覆盖范围与已知边界

- **DeepEP V2**：`tests/elastic/test_ep.py`，NCCL GIN / EFA-GDA type-5 路径，含 official 与 PR#1+PR#2 优化对比三组。
- **DeepEP V1（可选项）**：legacy NVSHMEM 路线。EFA 上 V1 无法用原版代码（upstream NVSHMEM 无 EFA 传输层、internode kernel 硬依赖 IBGDA），采用配对 fork 构建独立镜像（`amazon-contributing/upstream-to-nvshmem` + `rauteric/DeepEP`），B300 上另需 fork 的 Kineto 计时兜底分支（B300 上 profiler 返回 0 事件，原版测试脚本会崩）。覆盖 test_intranode（smoke）/ test_internode（Normal 模式）/ test_low_latency（Low Latency 模式），官方默认参数。测试开始前由用户选择是否启用（`V1_ENABLED`）。
- NCCL 覆盖 allreduce / alltoall / allgather / reducescatter 四个原语（官方 nccl-tests，容器内 mpirun）。
- 驱动脚本当前固定 2 节点；4 节点需按 TESTPLAN §4.2 的说明扩展。
- 仅支持 Ubuntu 24.04 DL Base OSS Nvidia Driver GPU AMI（须支持 B300）+ p6-b300。

## B300 与 p5en 的硬差异（本工具包已内置处理）

| | p5en.48xlarge | p6-b300.48xlarge |
|---|---|---|
| 网卡布局 | 16 张卡，card0 = `efa` | **17 张卡**，card0 **只能是 ENA**，card1..16 = `efa-only` |
| 每张 EFA / 节点聚合 | 200 Gb/s / 3.2 Tbps | **400 Gb/s / 6.4 Tbps**（每 GPU 线速 100 GB/s） |
| ibverbs 设备 | 16 个，全是 EFA | **18 个**（16 `rdmap*` + 2 非 EFA），必须 `NCCL_IB_HCA=rdmap`（已烧进镜像） |
| V2 构建 | sm90 / CUDA 13.0.2 | **sm103 / CUDA 13.3.1**（13.0.2 的 ptxas 在第一次 dispatch 才炸） |
| V1 计时 | Kineto 正常 | Kineto 返回 0 事件，钉带兜底的分支 |
| 子网 IP 消耗 | 每台 16 个 | 每台 1 个（efa-only 不占 IPv4） |

注意 17（网卡）和 18（ibverbs 设备）不矛盾——两套视角：card0 的 ENA 不是 RDMA 设备，不出现在 `ibv_devinfo -l` 里；那 2 个 `ibp*` 是 B300 机型恒有的板载非 EFA RDMA 设备，与 17 张 ENI 无对应。1+16=17 张卡，16+2=18 个 ibverbs 设备，各自都对。

## 目录

```text
TESTPLAN.md                           执行手册（主文档）
docker/Dockerfile                     DeepEP V2 + nccl-tests 测试镜像（自包含构建，sm103）
docker/Dockerfile.v1                  DeepEP V1 测试镜像（可选项）
scripts/generate_launch_template.py   17 网卡（16 EFA）启动模板生成
scripts/run_deepep_case.sh            DeepEP V2 双节点 case 驱动
scripts/run_deepep_v1_case.sh         DeepEP V1 双节点 case 驱动
scripts/run_nccl_case.sh              NCCL case 驱动
scripts/parse_deepep.py               DeepEP V2 结果解析与轮间一致性检查
scripts/parse_deepep_v1.py            DeepEP V1 结果解析
```

## 参考数字（未用本工具包完整复测，仅供量级核对）

本工具包尚未在 B300 上完整跑过一轮（跑完欢迎提 PR 补充 examples/）。以下参考值来自同软件栈的前次实测，出处标注在各表：

**DeepEP V2**（2×p6-b300，2026-08-29，[terrificdm/p6b300-efa-deepepv2-test-runbook](https://github.com/terrificdm/p6b300-efa-deepepv2-test-runbook) 实测；形状与本工具包一致：8192/128 tok、hidden 7168、top-8、256 experts、FP8 dispatch、type-5，16 rank 全 rank 均值）：

| 场景（12 SM，同本工具包默认） | official | PR#1+#2 | PR#1+#2 + EP_NUM_SUB_PARTS=1 |
|---|--:|--:|--:|
| decode (128 tok) dispatch 时延 | 288.7 µs | 130.1 µs (−54.9%) | 129.6 µs (−55.1%) |
| decode combine 时延 | 165.6 µs | 165.3 µs | ≈持平 |
| prefill (8192 tok) dispatch 时延 | 1039.5 µs | ≈持平 | — |
| prefill combine 时延 | 2878.5 µs | ≈持平 | — |

要点：
- **decode 必须叠加 PR#1+#2**：官方基线的 decode dispatch 在 B300 上有回归（比同形状 p5en 慢约 1.7 倍），PR 打上后省 55%、基本追平 p5en；`EP_NUM_SUB_PARTS=1` 在 B300 上接近中性（p5en 上还能再省几个点）。
- **prefill 对 SM 数敏感**：24 SM 档 dispatch 886 µs / combine 1837 µs，比 12 SM 快 15%/36%——24 SM 加测在 B300 上很值得做（改一个参数即可，见 TESTPLAN §4.2）。
- 绝对带宽约为 p5en 的 1.7 倍（每卡 200→400 Gb/s），但线速利用率反而略低（prefill 24 SM 约 69% vs p5en 约 80%）。

**DeepEP V1**（可选项，官方默认参数，与 V2 口径不同不可直接对比；[whn09/ep-benchmarks-efa](https://github.com/whn09/ep-benchmarks-efa) B300 参考实测）：Normal 模式 dispatch 109.8 (BF16) / combine 101.7 GB/s（RDMA）——EFA 上首次两个方向都超过 100 GB/s，约为官方 IB 参考（43 GB/s）的 2.5 倍；Low Latency 模式时延与 p5en 量级相近（CPU proxy 路径是瓶颈，不随网卡换代明显变化）。

**NCCL**：B300 上暂无实测参考。量级判据：每节点网络聚合 800 GB/s（p5en 的 2 倍，p5en 上 8 GiB busbw 为 allreduce 474.7 / alltoall 87.7 GB/s），大消息 busbw 只有个位数即为回落 TCP，数字作废。

## 致谢与来源

- 镜像构建依据 AWS 官方参考：[awslabs/awsome-distributed-ai](https://github.com/awslabs/awsome-distributed-ai)（deepep-v2-benchmark，MIT-0）
- B300 主机配置、17 网卡布局、sm103 构建组合与 V2 实测经验：[terrificdm/p6b300-efa-deepepv2-test-runbook](https://github.com/terrificdm/p6b300-efa-deepepv2-test-runbook)
- V1/V2 B300 镜像构建方法参考：[whn09/ep-benchmarks-efa](https://github.com/whn09/ep-benchmarks-efa)（MIT）
- 被测对象：[amazon-contributing/DeepEP](https://github.com/amazon-contributing/DeepEP)（AWS 指定的 EFA fork，V2）、[rauteric/DeepEP](https://github.com/rauteric/DeepEP) + [amazon-contributing/upstream-to-nvshmem](https://github.com/amazon-contributing/upstream-to-nvshmem)（V1 配对 fork）、[NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)
- 同系列 p5en 版工具包：[terrificdm/p5en-efa-deepep-nccl-testkit](https://github.com/terrificdm/p5en-efa-deepep-nccl-testkit)

## License

[Apache-2.0](LICENSE)
