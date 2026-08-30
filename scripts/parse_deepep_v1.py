#!/usr/bin/env python3
"""DeepEP V1 日志解析：Normal 模式(internode) 取自动调优 Best 行，Low Latency 模式取全 rank 均值。

用法: python3 parse_deepep_v1.py run/logs
输出: 每个 v1-* tag 的关键指标 + 同名 r1/r2 轮间偏差表(>5% 标 RETRY)。
口径:
  - internode: leader 日志的 [tuning] Best dispatch (FP8/BF16) / Best combine 行,
    RDMA GB/s 是跨节点 EFA 带宽,NVL GB/s 是节点内 NVLink 带宽
  - low_latency: leader+worker 全 16 rank 的 avg_t/带宽均值
    (dispatch、combine 分项 + dispatch+combine 合并)
  - B300 注意: 日志若含 "Kineto profiler returned 0 events"（镜像自带兜底），
    low_latency 的 dispatch/combine 分项是合计均摊值，只有 dispatch+combine 合计准确，
    报告里注明口径
"""
import glob
import json
import re
import sys


def ll_stats(logdir, tag):
    disp, comb, both = [], [], []
    for f in glob.glob(f'{logdir}/{tag}-*.log'):
        for line in open(f):
            m = re.search(r'Dispatch bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us \| '
                          r'Combine bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us', line)
            if m:
                disp.append((float(m[2]), float(m[1])))
                comb.append((float(m[4]), float(m[3])))
            m2 = re.search(r'Dispatch \+ combine bandwidth: ([\d.]+) GB/s, avg_t=([\d.]+) us', line)
            if m2:
                both.append((float(m2[2]), float(m2[1])))
    if not disp:
        return None
    avg = lambda xs, i: round(sum(x[i] for x in xs) / len(xs), 2)
    return {'ranks': len(disp),
            'dispatch': {'us': avg(disp, 0), 'gbps': avg(disp, 1)},
            'combine': {'us': avg(comb, 0), 'gbps': avg(comb, 1)},
            'dispatch_combine': {'us': avg(both, 0), 'gbps': avg(both, 1)}}


def normal_stats(logdir, tag):
    out = {}
    key_map = {'dispatch (FP8)': 'dispatch_fp8', 'dispatch (BF16)': 'dispatch_bf16', 'combine': 'combine'}
    try:
        lines = open(f'{logdir}/{tag}-leader.log').readlines()
    except FileNotFoundError:
        return None
    for line in lines:
        m = re.search(r'Best (dispatch \(FP8\)|dispatch \(BF16\)|combine).*?'
                      r'([\d.]+) GB/s \(RDMA\), ([\d.]+) GB/s \(NVL\)', line)
        if m:
            out[key_map[m[1]]] = {'rdma_gbps': float(m[2]), 'nvl_gbps': float(m[3])}
    return out or None


def main(logdir):
    tags = sorted({re.sub(r'-(leader|worker)\.log$', '', f.split('/')[-1])
                   for f in glob.glob(f'{logdir}/v1-*.log')})
    result = {}
    for tag in tags:
        s = ll_stats(logdir, tag) if 'lowlatency' in tag else normal_stats(logdir, tag)
        if s:
            result[tag] = s
    print(json.dumps(result, indent=1))

    print('\n轮间一致性 (>5% => RETRY):', file=sys.stderr)
    for base in sorted({t[:-3] for t in result if re.search(r'-r[12]$', t)}):
        r1, r2 = result.get(base + '-r1'), result.get(base + '-r2')
        if not (r1 and r2):
            continue
        if 'lowlatency' in base:
            a, b = r1['dispatch_combine']['us'], r2['dispatch_combine']['us']
        else:
            a, b = r1['combine']['rdma_gbps'], r2['combine']['rdma_gbps']
        dev = abs(a - b) / min(a, b) * 100
        print(f'  {base}: r1={a} r2={b} 偏差={dev:.1f}% {"OK" if dev <= 5 else "RETRY"}',
              file=sys.stderr)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'run/logs')
