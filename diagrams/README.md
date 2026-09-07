# 架构图

本目录是 [TESTPLAN.md](../TESTPLAN.md) 的配图，六张图覆盖从主机准备到测试执行的完整链路。每个 HTML 都是自包含单文件（内联样式与脚本），支持深浅色切换。

> **点下面「查看」栏的链接**即可在浏览器中打开渲染后的图。HTML 源文件不在 `main`，而是放在 [`gh-pages`](https://github.com/terrificdm/p6b300-efa-deepep-nccl-testkit/tree/gh-pages/diagrams) 分支上由 GitHub Pages 托管——每个文件约 710 KB 且其中约 96% 是渲染运行时，留在 `main` 会显著拖累仓库的克隆体积与 AI agent 的上下文预算。

| # | 图 | 内容 | 查看 |
|---|---|---|---|
| 01 | 总体架构 | P6-B300 多节点测试环境总体架构（N = 2 或 4 · EFA / DeepEP / NCCL） | [打开](https://terrificdm.github.io/p6b300-efa-deepep-nccl-testkit/diagrams/01-overview.html) |
| 02 | 主机准备流程 | 网络 → P6-B300 → GDRCopy / EFA installer → EFA-GDA 配置 | [打开](https://terrificdm.github.io/p6b300-efa-deepep-nccl-testkit/diagrams/02-host-prep.html) |
| 03 | 容器镜像与运行 | V2+NCCL 共用镜像 / V1 独立镜像（EFA 用户态与 GDRCopy 分开安装） | [打开](https://terrificdm.github.io/p6b300-efa-deepep-nccl-testkit/diagrams/03-container-build-run.html) |
| 04 | EFA 软件栈与数据路径 | DeepEP V2（GIN type 5）/ NCCL（OFI net）/ DeepEP V1（NVSHMEM） | [打开](https://terrificdm.github.io/p6b300-efa-deepep-nccl-testkit/diagrams/04-efa-stack-datapath.html) |
| 05 | 测试拉起时序 | DeepEP 临时容器 + torchrun；NCCL 常驻容器 + mpirun（N=2/4） | [打开](https://terrificdm.github.io/p6b300-efa-deepep-nccl-testkit/diagrams/05-test-launch-sequence.html) |
| 06 | 软件版本与测试矩阵 | 软件版本、镜像变体与测试矩阵 | [打开](https://terrificdm.github.io/p6b300-efa-deepep-nccl-testkit/diagrams/06-versions-test-matrix.html) |

**[📁 在线目录浏览](https://terrificdm.github.io/p6b300-efa-deepep-nccl-testkit/diagrams/)** —— 一个页面列出全部六张图。

## 本地查看

```bash
git fetch origin gh-pages && git worktree add ../diagrams-gh gh-pages
open ../diagrams-gh/diagrams/01-overview.html      # macOS
xdg-open ../diagrams-gh/diagrams/01-overview.html  # Linux
```

单文件无外部依赖，下载后离线也能打开（仅正文等宽字体走 Google Fonts CDN，取不到时回退系统字体，不影响图形与内容）。
