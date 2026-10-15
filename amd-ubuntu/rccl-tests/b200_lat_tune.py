#!/usr/bin/env python3
"""Result of b200-nodes/job-nccl-lat-2node-tune.sh (B200, 2 nodes x 8 GPUs, fixed NCCL_ALGO / NCCL_PROTO).

  b200_lat_tune.py <b200-nodes/out-nccl-lat-tune/JOB>

Question: is B200's slow 1 MB latency across 2 nodes (which gives MI355X its 1 MB wins) an NCCL
setting? Writes
  <JOB>/summary.md                                   full table, all settings
  results/mi355x-vs-b200-interconnect.md             block between the B200-TUNE markers (§2.3)
  results/mi355x-vs-b200.md + interconnect Quick summary: the last bullet of the inter-node latency row
MI355X = ROCm 7.14, the faster of default and NCCL_NET=IB (as in compare_lat.py).
"""
import glob, os, re, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_lat import load_runs, newest_done, LOGS  # noqa: E402

BENCH = "/orcd/data/orcd/022/benchmarks/amd-ubuntu"
RES = os.environ.get("TUNE_RES", f"{BENCH}/results")   # override for tests
MAIN = f"{RES}/mi355x-vs-b200.md"
IC = f"{RES}/mi355x-vs-b200-interconnect.md"
BEGIN, END = "<!-- B200-TUNE-BEGIN (auto: rccl-tests/b200_lat_tune.py) -->", "<!-- B200-TUNE-END -->"
OLD = "/orcd/data/orcd/022/benchmarks/b200-nodes/out-nccl-lat/25690497"
COLLS = ["all_gather", "reduce_scatter", "broadcast", "alltoall", "all_reduce", "sendrecv"]
FOUR = ["all_gather", "reduce_scatter", "broadcast", "alltoall"]
SIZES = [256, 65536, 1 << 20, 4 << 20]
NAMES = {"ll": "NCCL_PROTO=LL", "ll128": "NCCL_PROTO=LL128", "simple": "NCCL_PROTO=Simple",
         "ring": "NCCL_ALGO=Ring", "ringll": "NCCL_ALGO=Ring NCCL_PROTO=LL",
         "ringll128": "NCCL_ALGO=Ring NCCL_PROTO=LL128", "ringsimple": "NCCL_ALGO=Ring NCCL_PROTO=Simple",
         "tree": "NCCL_ALGO=Tree", "treell": "NCCL_ALGO=Tree NCCL_PROTO=LL", "nvls0": "NCCL_NVLS_ENABLE=0"}
ALGO = {0: "Tree", 1: "Ring", 2: "CollNetDirect", 3: "CollNetChain", 4: "NVLS", 5: "NVLSTree", 6: "PAT"}
PROTO = {0: "LL", 1: "LL128", 2: "Simple"}
unit = lambda s: {256: "256 B", 65536: "64 KB", 1 << 20: "1 MB", 4 << 20: "4 MB"}[s]
f1 = lambda x: "–" if x is None else f"{x:.1f}"
fx = lambda x: "–" if x is None else f"{x:.2f}x"


def rng(v, f="{:.1f}"):
    a, b = f.format(min(v)), f.format(max(v))
    return a if a == b else f"{a}–{b}"


def main():
    tdir = sys.argv[1].rstrip("/")
    job = os.path.basename(tdir)
    if "latency run DONE" not in open(os.path.join(tdir, "STATE.txt")).read():
        sys.exit("run not finished: " + tdir)
    t = load_runs(tdir)
    m = load_runs(newest_done(f"{LOGS}/rocm7.14/rccl/lat_2node_*"))
    old = load_runs(OLD)
    g = lambda r, k, c, s: r.get((k, "2n_ppn8"), {}).get(c, {}).get(s)

    def bdef(c, s):  # B200 default = mean of the two default runs of this job
        v = [x for x in (g(t, "r1", c, s), g(t, "r2", c, s)) if x]
        return statistics.mean(v) if v else None

    def bbest(c, s):
        v = [(g(t, k, c, s), k) for k in NAMES if g(t, k, c, s)]
        return min(v) if v else (None, None)

    def mbest(c, s):
        v = [x for x in (g(m, "default", c, s), g(m, "netib", c, s)) if x]
        return min(v) if v else None

    # which algorithm / protocol NCCL's default picks (TUNING log), at 1 MB
    picks = {}
    for c in ("all_gather", "reduce_scatter", "broadcast", "all_reduce"):
        f = os.path.join(tdir, f"tuning_{c}.log")
        for line in open(f, errors="replace") if os.path.exists(f) else []:
            q = re.search(r"\b1048576 Bytes -> Algo (\d+) proto (\d+)", line)
            if q:
                picks[c] = f"{ALGO.get(int(q.group(1)), q.group(1))}/{PROTO.get(int(q.group(2)), q.group(2))}"
                break

    # summary.md in the job dir: every setting
    D = [f"# B200 NCCL latency with fixed settings, 2 nodes × 8 GPUs (job {job})", "",
         "Time per call in µs (lower is better), nccl-tests 1000 iterations + 100 warm-up. Default = mean of 2 default runs.", ""]
    keys = ["r1", "r2"] + list(NAMES)
    D += ["| collective | bytes | " + " | ".join("default " + k[1] if k in ("r1", "r2") else NAMES[k] for k in keys) + " |",
          "|---|---:|" + "---:|" * len(keys)]
    for c in COLLS:
        for s in SIZES:
            D.append(f"| {c} | {unit(s)} | " + " | ".join(f1(g(t, k, c, s)) for k in keys) + " |")
    D += ["", "Default picks at 1 MB (TUNING log): " + (", ".join(f"{c} {p}" for c, p in picks.items()) or "not found in the log") + "."]
    open(os.path.join(tdir, "summary.md"), "w").write("\n".join(D) + "\n")

    # the 1 MB question
    rows, ok, part, newr, oldr = [], 0, 0, [], []
    for c in COLLS:
        for s in (65536, 1 << 20, 4 << 20):
            y, (yb, k), x = bdef(c, s), bbest(c, s), mbest(c, s)
            if not (y and x):
                continue
            rows.append(f"| {c} | {unit(s)} | {f1(g(old, 'default', c, s))} | {f1(y)} | {f1(yb)} | "
                        f"{NAMES.get(k, '–') if yb and yb < 0.97 * y else 'none faster than default'} | {f1(x)} | {fx(y / x)} | {fx(yb / x) if yb else '–'} |")
            if c in FOUR and s == 1 << 20 and yb:
                oldr.append(y / x); newr.append(yb / x)
                ok += yb / x <= 1.03
                part += yb <= 0.8 * y
    n = len(newr)
    if n and ok == n:
        v = (f"**Confirmed: it is an NCCL setting.** With a fixed setting B200's 1 MB time drops, and MI355X's 1 MB wins in "
             f"all_gather, reduce_scatter, broadcast and alltoall go away ({rng(oldr, '{:.2f}')}x with NCCL's default, "
             f"{rng(newr, '{:.2f}')}x with B200's best fixed setting).")
        bullet = "MI355X faster in those at 1 MB only with NCCL's default; with a fixed NCCL setting B200 is as fast or faster"
    elif part:
        v = (f"**Partly confirmed.** A fixed setting makes B200 faster at 1 MB in {part} of {n} collectives, but MI355X stays "
             f"faster in {n - ok} of them ({rng(oldr, '{:.2f}')}x with NCCL's default, {rng(newr, '{:.2f}')}x with B200's best fixed setting).")
        bullet = f"MI355X {rng(newr, '{:.1f}')}× faster in those at 1 MB, also with B200's best fixed NCCL setting (default: {rng(oldr, '{:.1f}')}×)"
    else:
        v = (f"**Not confirmed: no fixed setting helps.** B200's best fixed setting stays within 20% of the default at 1 MB "
             f"({rng(newr, '{:.2f}')}x vs {rng(oldr, '{:.2f}')}x), so the 1 MB gap is not an NCCL algorithm or protocol choice.")
        bullet = f"MI355X {rng(newr, '{:.1f}')}× faster in those at 1 MB, also with B200's best fixed NCCL setting"
    rep = [abs(bdef(c, 1 << 20) / g(old, "default", c, 1 << 20) - 1) for c in FOUR
           if bdef(c, 1 << 20) and g(old, "default", c, 1 << 20)]
    same = (f" This job's default reproduces the earlier run within {100 * max(rep):.0f}%." if rep and max(rep) <= 0.15 else
            f" Note: this job's default differs from the earlier run by up to {100 * max(rep):.0f}% (other nodes or load)." if rep else "")
    blk = [BEGIN,
           f"- **1 MB, B200 rerun with fixed `NCCL_ALGO` / `NCCL_PROTO` (job {job}):** {v}{same}"
           + (f" NCCL's default at 1 MB: {', '.join(f'{c} {p}' for c, p in picks.items())}." if picks else ""), "",
           "  | collective | bytes | B200 default, earlier run | B200 default, this job | B200 best fixed | best setting | MI355X | MI355X / B200 speed, default | MI355X / B200 speed, B200 best fixed |",
           "  |---|---:|---:|---:|---:|---|---:|---:|---:|"] + ["  " + r for r in rows] + [
           "", f"  Time per call, µs. All settings: `../../b200-nodes/out-nccl-lat-tune/{job}/summary.md`.", END]
    s = open(IC).read()
    s = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: "\n".join(blk), s, flags=re.S)
    old_b = "MI355X 1.2–1.7× faster in those at 1 MB, likely an NCCL tuning issue on B200"
    s = s.replace(old_b, bullet)
    open(IC, "w").write(s)
    s = open(MAIN).read().replace(old_b, bullet)
    open(MAIN, "w").write(s)
    print("updated", IC, MAIN, os.path.join(tdir, "summary.md"))
    print(v)


if __name__ == "__main__":
    main()
