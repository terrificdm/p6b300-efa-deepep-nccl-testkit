# gh-pages —— 仅存放架构图 HTML

这个分支只为 GitHub Pages 托管而存在，**不包含任何代码或文档**，也不与 `main` 共享历史。

- **看图**：<https://terrificdm.github.io/p6b300-efa-deepep-nccl-testkit/diagrams/>
- **图的索引与说明**：`main` 分支的 [`diagrams/README.md`](https://github.com/terrificdm/p6b300-efa-deepep-nccl-testkit/blob/main/diagrams/README.md)
- **代码与执行手册**：[`main`](https://github.com/terrificdm/p6b300-efa-deepep-nccl-testkit)

## 为什么单独开分支

每个图 HTML 约 710 KB，其中约 96% 是 archify 的渲染运行时（六个文件里逐字节相同），真实图形 SVG 只占 3.7%。把它们放在 `main` 会让仓库的 token 体量从约 60 K 涨到约 1.7 M，超出常见 1M 上下文窗口，妨碍 AI agent 阅读本仓库。放在这里，`main` 保持精简，Pages 照常工作。

## 更新图

替换 `diagrams/` 下对应的 HTML 后 commit push 即可，Pages 会自动重新部署。新增或删除图时，记得同步更新 `diagrams/index.html` 和 `main` 分支的 `diagrams/README.md`。
