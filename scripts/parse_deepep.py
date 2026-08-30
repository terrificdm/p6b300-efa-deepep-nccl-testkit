#!/usr/bin/env python3
"""解析 DeepEP case 日志：聚合全 rank 均值，输出 dispatch/combine/reduced-combine 的时延与 SO 带宽。"""
import re, sys, glob, os, json

PAT = {
    'dispatch': re.compile(r'^\s+\* EP:\s+\d+/\d+ \| dispatch: (\d+) GB/s \(SO\), (\d+) GB/s \(SU\), ([\d.]+) us'),
    'combine':  re.compile(r'^\s+@ EP:\s+\d+/\d+ \| combine:\s+(\d+) GB/s \(SO\), (\d+) GB/s \(SU\), ([\d.]+) us'),
    'reduced':  re.compile(r'^\s+\+ EP:\s+\d+/\d+ \| reduced combine: (\d+) GB/s \(SO\), (\d+) GB/s \(SU\), ([\d.]+) us'),
}

def parse_tag(logdir, tag):
    vals = {k: [] for k in PAT}
    files = [f'{logdir}/{tag}-leader.log', f'{logdir}/{tag}-worker.log']
    for f in files:
        if not os.path.exists(f): return None
        for line in open(f):
            for k, p in PAT.items():
                m = p.match(line)
                if m: vals[k].append((int(m[1]), int(m[2]), float(m[3])))
    out = {}
    for k, v in vals.items():
        if v:
            n = len(v)
            out[k] = {'ranks': n,
                      'so_gbps': round(sum(x[0] for x in v)/n, 1),
                      'su_gbps': round(sum(x[1] for x in v)/n, 1),
                      'us': round(sum(x[2] for x in v)/n, 2)}
    return out

def main():
    logdir = sys.argv[1] if len(sys.argv) > 1 else 'run/logs'
    tags = sorted({re.sub(r'-(leader|worker)\.log$', '', os.path.basename(f))
                   for f in glob.glob(f'{logdir}/*-r[12]-*.log')})
    results = {}
    for tag in tags:
        r = parse_tag(logdir, tag)
        if r: results[tag] = r
    print(json.dumps(results, indent=1, ensure_ascii=False))
    # 轮间一致性
    print('\n=== 轮间一致性 (dispatch us, r1 vs r2) ===')
    bases = sorted({t[:-3] for t in results if t.endswith(('-r1','-r2'))})
    for b in bases:
        a, c = results.get(b+'-r1'), results.get(b+'-r2')
        if a and c and 'dispatch' in a and 'dispatch' in c:
            u1, u2 = a['dispatch']['us'], c['dispatch']['us']
            dev = abs(u1-u2)/min(u1,u2)*100
            best = 'r1' if u1 <= u2 else 'r2'
            print(f'{b:24s} r1={u1:9.2f}us r2={u2:9.2f}us dev={dev:4.1f}% best={best} {"OK" if dev<=5 else "RETRY!"}')

main()
