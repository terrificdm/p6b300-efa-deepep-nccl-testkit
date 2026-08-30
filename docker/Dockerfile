# DeepEP V2 on AWS EFA —— p6-b300 (B300 / sm103) 专用镜像
#
# 依据两份公开材料合并而成：
#   1. AWS 官方参考镜像（MIT-0）：
#      awslabs/awsome-distributed-ai
#        micro-benchmarks/expert-parallelism/deepep-v2-benchmark/deepep.Dockerfile
#   2. B300 上已验证的构建组合：terrificdm/p6b300-efa-deepepv2-test-runbook 引用的
#      whn09/ep-benchmarks-efa deepep-v2-efa-official Dockerfile（sm103 档：
#      CUDA 13.3.1 + TORCH_CUDA_ARCH_LIST=10.3，2026-08-29 两台 p6-b300 实测跑通）
#
# 与 p5en 版（同系列 p5en-efa-deepep-nccl-testkit）的差异只有三处：
#   - CUDA base 13.0.2 -> 13.3.1（sm_103 硬下限，见下）
#   - TORCH_CUDA_ARCH_LIST 9.0 -> 10.3
#   - 运行时默认值：EP_NIC_NAME 换 b300 命名、新增 NCCL_IB_HCA=rdmap
#
# 为什么 CUDA 必须 >= 13.3.x：sm_103 会进 deep_ep/include/deep_ep/common/ptx.cuh
# 里多处 `#if __CUDA_ARCH__ >= 1000` 分支（st.bulk、.v4.s64 LD/ST），13.0.2 的
# ptxas 不认这些指令形式，报 `Arguments mismatch for instruction 'mov'` ->
# `ptxas fatal` -> `compiler.hpp:239 NVCC compilation failed`。而且是在第一次
# dispatch（JIT 触发点）才炸，build 阶段完全看不出来。没有宏能绕：JIT 自己在
# csrc/jit/compiler.hpp 拼 flags，没有往 JIT 传 -D 的入口。
#
# 分工：容器只装用户态（libfabric / rdma-core / aws-ofi-nccl / gdrcopy 库）。
# 内核态（efa.ko >= 3.3.0、efa_nv_peermem、gdrdrv）必须由宿主机提供。
#
# 依赖链（AWS 官方 README 的版本约束，缺一层 GIN 就起不来）：
#   DeepEP V2 kernels
#     -> NCCL GIN 设备 API      : nvidia-nccl >= 2.31 (nccl_device.h)
#     -> ncclGinPlugin_v14      : aws-ofi-nccl（EFA installer >= 1.50 附带）
#     -> EFA-GDA (GIN type 5)   : libfabric efa-direct
#     -> completion counter API : 宿主机 efa.ko >= 3.3.0
#
# 构建（需先把 EFA installer tarball 放到本目录）：
#   docker build -t deepep-v2-efa:official .

ARG CUDA_VERSION=13.3.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git pkg-config \
        python3 python3-dev python3-pip python3-venv \
        libhwloc-dev pciutils environment-modules kmod check libsubunit-dev \
        openssh-client openssh-server \
    && rm -rf /var/lib/apt/lists/*

# 删掉 CUDA 基础镜像自带的 apt NCCL（2.28.x，无 GIN）。留着它的话，
# 任何人动一下 LD_LIBRARY_PATH 就会静默掉到没有 GIN 的旧版本上。
RUN apt-mark unhold libnccl2 libnccl-dev 2>/dev/null || true \
    && apt-get remove -y libnccl2 libnccl-dev 2>/dev/null || true \
    && ldconfig

# ---------------------------------------------------------------------------
# 1) EFA 用户态（与宿主机使用同一个已校验的 tarball）
#    --skip-kmod       容器里不装内核模块
#    --skip-limit-conf memlock 由 docker --ulimit 提供
#    最后的 nm 检查是关键门禁：aws-ofi-nccl 必须导出 ncclGinPlugin_v14
#    （EFA-GDA 的 op table）。没有 v14 的镜像照样能构建、能跑，
#    但会静默走 CPU proxy，测出来的不是 GPU 直发的性能。
# ---------------------------------------------------------------------------
ARG EFA_INSTALLER_VERSION=1.50.0
ENV EP_EFA_INSTALLER=${EFA_INSTALLER_VERSION}
COPY aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz /tmp/aws-efa-installer.tar.gz
RUN apt-get update && cd /tmp && tar xzf aws-efa-installer.tar.gz \
    && { grep -q "^## \[${EFA_INSTALLER_VERSION}\]" aws-efa-installer/ChangeLog.md \
         || { echo "tarball is not EFA installer ${EFA_INSTALLER_VERSION}" >&2; exit 1; }; } \
    && cd aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && ldconfig \
    && { nm -D /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so | grep -qw ncclGinPlugin_v14 \
         || { echo "aws-ofi-nccl exports no ncclGinPlugin_v14 => no EFA-GDA" >&2; exit 1; }; } \
    && cd / && rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

ENV PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/amazon/efa/lib:/opt/amazon/ofi-nccl/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH

# ---------------------------------------------------------------------------
# 2) GDRCopy 用户态库（GIN 初始化要开 gdr handle；gdrdrv 内核模块在宿主机）
# ---------------------------------------------------------------------------
RUN git clone -b v2.5.2 --depth 1 https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy \
    && cd /tmp/gdrcopy && make prefix=/usr/local lib_install \
    && ldconfig && cd / && rm -rf /tmp/gdrcopy

# ---------------------------------------------------------------------------
# 3) PyTorch + NCCL(pip) + NVSHMEM
#    NCCL >= 2.31：GIN 设备 API（nccl_device.h）
#    NVSHMEM：DeepEP setup.py 的硬依赖（legacy 后端），只跑 V2 也要装
#    torch wheel 是 cu130，CUDA 次版本兼容，13.3.1 base 不用换 wheel——
#    CUDA 版本只决定 JIT 用的 ptxas/nvcc（b300 实测验证过这套组合）
# ---------------------------------------------------------------------------
RUN pip3 install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0+cu130 \
    && pip3 install --break-system-packages --no-cache-dir \
        "nvidia-nccl-cu13==2.31.2" "nvidia-nvshmem-cu13==3.7.2" --no-deps \
    && pip3 install --break-system-packages --no-cache-dir \
        ninja numpy nvidia-ml-py

# ---------------------------------------------------------------------------
# 3b) nccl-tests（NVIDIA 官方，MPI 版）—— NCCL 测试也在容器里跑
#     链接到上面 pip 装的同一份 NCCL 2.31.2，保证 nccl-tests 测的
#     和 DeepEP 用的是同一个 NCCL。
#     /opt/nccl 是指向 pip NCCL 包的固定路径符号链接，方便 ENV 引用。
# ---------------------------------------------------------------------------
RUN NCCL_ROOT="$(python3 -c 'import nvidia.nccl;print(list(nvidia.nccl.__path__)[0])')" \
    && ln -s "$NCCL_ROOT" /opt/nccl \
    && ln -sf /opt/nccl/lib/libnccl.so.2 /opt/nccl/lib/libnccl.so \
    && git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && make -j MPI=1 MPI_HOME=/opt/amazon/openmpi \
         CUDA_HOME=/usr/local/cuda NCCL_HOME=/opt/nccl \
    && test -x build/all_reduce_perf
ENV LD_LIBRARY_PATH=/opt/nccl/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/nccl-tests/build:$PATH

# ---------------------------------------------------------------------------
# 3c) 容器内 sshd（跨节点 mpirun 用：leader 容器 ssh 到 worker 容器拉起 orted）
#     监听 2222（host 网络下 22 被宿主机 sshd 占用）；密钥在运行时注入，
#     不烧进镜像。MPI 控制面走 TCP，数据面由 NCCL 走 EFA，互不相干。
# ---------------------------------------------------------------------------
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && printf 'Host *\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  Port 2222\n' >> /etc/ssh/ssh_config

# Open MPI 控制面设置（与 AWS 官方参考镜像一致）
ENV OMPI_MCA_pml=^ucx \
    OMPI_MCA_btl=tcp,self \
    OMPI_MCA_btl_tcp_if_exclude=lo,docker0,veth_def_agent \
    OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo,veth \
    PMIX_MCA_gds=hash

# ---------------------------------------------------------------------------
# 4) DeepEP V2（amazon-contributing fork —— AWS 官方指定的 EFA 版本，
#    含 EFA GDAKI 所需的 unordered kernels；官方测试脚本 tests/ 原样在内）
#    10.3 = Blackwell Ultra（p6-b300 / B300）。单架构镜像：Hopper 的 cubin
#    在 Blackwell 上跑不了，反之亦然；且 sm_103 的 CUDA base 下限（13.3.x）
#    和 sm_90（13.0.2 即可）不同，"一个镜像通吃两代"本来就得连 base 一起换。
# ---------------------------------------------------------------------------
ARG TORCH_CUDA_ARCH_LIST=10.3
ARG DEEPEP_REF=9c1f251118873e5dee9652cdfaf9848753cda7f1
ARG CUDA_VERSION
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    EP_BUILD_ARCH=${TORCH_CUDA_ARCH_LIST} \
    EP_BUILD_CUDA=${CUDA_VERSION} \
    EP_JIT_CACHE_DIR=/root/.deep_ep
RUN mkdir -p /opt/DeepEP && cd /opt/DeepEP && git init -q \
    && git remote add origin https://github.com/amazon-contributing/DeepEP.git \
    && git fetch -q --depth 1 origin "${DEEPEP_REF}" \
    && git checkout -q FETCH_HEAD \
    && git rev-parse HEAD > /opt/DeepEP/BUILD_REF \
    && git submodule update -q --init --recursive --depth 1 \
    && NCCL_ROOT="$(python3 -c 'import nvidia.nccl;print(list(nvidia.nccl.__path__)[0])')" \
    && test -f "$NCCL_ROOT/include/nccl_device.h" \
    && EP_NCCL_ROOT_DIR="$NCCL_ROOT" \
       LIBRARY_PATH="$NCCL_ROOT/lib:${LIBRARY_PATH:-}" \
       pip3 install --break-system-packages --no-build-isolation -v . \
    && rm -rf build

# ---------------------------------------------------------------------------
# 5) 运行时默认值
#    EP_NIC_NAME：EFA 上网卡名不是 mlx5_0；b300 实测默认 rdmap101s0，
#    实际名字用 `ibv_devinfo -l` 查，启动容器时按需 -e 覆盖。
#    NCCL_IB_HCA=rdmap：b300 特有且必须——该机型 ibverbs 恒有 18 个设备
#    （16 个 EFA rdmap* + 2 个非 EFA ibp*），不筛选的话 GIN 只建得出
#    2 个 GDAKI NIC 直接报错（"only 2 GIN GDAKI NICs have been created"）。
#    p5en 上 16 个设备全是 EFA 所以不需要；b300 上没有它跑不起来。
#    NCCL_GIN_TYPE=5 不在这里写死，由启动命令显式传入（见 TESTPLAN §4.2）。
# ---------------------------------------------------------------------------
ENV EP_NIC_NAME=rdmap101s0 \
    NCCL_IB_HCA=rdmap \
    NCCL_NET_PLUGIN=ofi \
    NCCL_TUNER_PLUGIN=ofi \
    FI_PROVIDER=efa \
    FI_EFA_USE_DEVICE_RDMA=1

# 不要把 WORKDIR 设成 /opt/DeepEP：python 会优先 import 源码目录里的
# deep_ep/（不含编译好的 _C.so），报 ModuleNotFoundError: deep_ep._C
WORKDIR /workspace
