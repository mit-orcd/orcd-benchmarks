#!/usr/bin/env python3
"""Pick the RCCL setting for 2-node all_reduce on ROCm 10.1 from run_2node_fix_rocm10.1.sh's check runs.

  pick_fix_rocm10.1.py <fix_2node dir>
Reads <dir>/check_runs.txt (tag|EXTRA_ENV|run dir), parses all_reduce_ppn8_np16.log in each run dir
(out-of-place time and #wrong per size). A setting is correct when no size has wrong results.
The fix = the correct setting with the lowest median time over 8 B-256 KB; RCCL's default is used
if it is correct. Writes results/node6100/rocm10.1/fix_choice.json and fix_check.md.
"""
import glob, json, os, re, statistics, sys

RES = "/orcd/data/orcd/022/benchmarks/amd-ubuntu/results/node6100/rocm10.1"
d = sys.argv[1]


def parse(f):
    """{size: (time_us, wrong)} from a rccl-tests log (out-of-place columns)."""
    rows = {}
    for line in open(f, errors="replace"):
        p = line.split()
        if len(p) >= 13 and p[0].isdigit() and p[1].isdigit() and p[0] != "0":
            try:
                t = float(p[5])
            except ValueError:
                continue
            w = sum(int(x) for x in (p[8], p[12]) if x.isdigit())
            rows[int(p[0])] = (t, w)
    return rows


def human(s):
    for u, k in (("M", 1 << 20), ("K", 1 << 10)):
        if s >= k and s % k == 0:
            return f"{s // k}{u}"
    return str(s)


res = []
for line in open(os.path.join(d, "check_runs.txt")):
    tag, env, rd = line.rstrip("\n").split("|")
    logs = glob.glob(os.path.join(rd, "all_reduce_ppn8_np16.log")) if rd else []
    rows = parse(logs[0]) if logs else {}
    wrong = sorted(s for s, (t, w) in rows.items() if w)
    small = [t for s, (t, w) in rows.items() if s <= 256 << 10]
    res.append(dict(tag=tag, env=env, rows=rows, wrong=wrong, ok=bool(rows) and not wrong,
                    med=statistics.median(small) if small else None))

good = [r for r in res if r["ok"] and r["med"]]
dflt = next((r for r in res if r["tag"] == "default"), None)
if dflt and dflt["ok"]:
    fix = dflt
else:
    fix = min(good, key=lambda r: r["med"]) if good else None
bad = [r["tag"] for r in res if not r["ok"]]
algo = ""
dbg = glob.glob(os.path.join(d, "debug_runs.txt"))
if dbg:
    rd = open(dbg[0]).read().split("|")[-1].strip()
    for f in glob.glob(os.path.join(rd, "all_reduce_*.log")):
        m = re.findall(r"AllReduce: opCount.*?algorithm (\d+) protocol (\d+)", open(f, errors="replace").read())
        if m:
            algo = f"algorithm {m[0][0]} protocol {m[0][1]}"
json.dump(dict(fix_env=(fix["env"] if fix else ""), fix_tag=(fix["tag"] if fix else ""),
               default_ok=bool(dflt and dflt["ok"]), bad_settings=bad,
               wrong_sizes_default=(dflt["wrong"] if dflt else []), algo_debug=algo,
               medians={r["tag"]: r["med"] for r in res}),
          open(f"{RES}/fix_choice.json", "w"), indent=1)

sizes = [8, 1024, 32768, 65536, 262144, 1 << 20, 4 << 20]
L = ["| RCCL setting | sizes with wrong results | " + " | ".join(human(s) for s in sizes) + " |",
     "|---|---|" + "---:|" * len(sizes)]
for r in res:
    name = f"`{r['env']}`" if r["env"] else "default"
    if fix and r is fix:
        name = f"**{name}** (used)"
    wr = ", ".join(human(s) for s in r["wrong"]) if r["wrong"] else ("none" if r["rows"] else "run failed")
    cells = []
    for s in sizes:
        t = r["rows"].get(s)
        cells.append("–" if not t else (f"{t[0]:,.1f}" + (" ✗" if t[1] else "")))
    L.append(f"| {name} | {wr} | " + " | ".join(cells) + " |")
open(f"{RES}/fix_check.md", "w").write("\n".join(L) + "\n")
print(f"fix={fix['env'] if fix else None!r} default_ok={bool(dflt and dflt['ok'])} bad={bad}")
