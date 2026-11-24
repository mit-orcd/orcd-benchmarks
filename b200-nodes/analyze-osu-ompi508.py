#!/usr/bin/env python3
"""Write <dir>/report.md from the newest osu-*.out in <dir>.

analyze-osu-ompi508.py                       -> out-osu-ib/ompi-5.0.8 (two openmpi/5.0.8 modules)
analyze-osu-ompi508.py ucx-net-devices "<title>" -> out-osu-ib/ucx-net-devices
"""
import glob, os, re, sys, time

SUB = sys.argv[1] if len(sys.argv) > 1 else "ompi-5.0.8"
TITLE = sys.argv[2] if len(sys.argv) > 2 else "OSU MPI p2p on B200 with the two `openmpi/5.0.8` modules"
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out-osu-ib", SUB)
LABEL = {"spack": "Spack gcc/12.2.0 `openmpi/5.0.8`", "core": "core `openmpi/5.0.8`",
         "ompi414": "default `openmpi/4.1.4`",
         "nvhpc-HH": "nvhpc/26.1 HPC-X OpenMPI, host buffers (H H)",
         "nvhpc-DD": "nvhpc/26.1 HPC-X OpenMPI, GPU buffers (D D, CUDA-aware)"}
JOBS = {"ompi-5.0.8": "job-osu-ompi508.sh", "ucx-net-devices": "job-osu-ucxdev.sh",
        "nvhpc-26.1": "job-osu-nvhpc261.sh"}
LINE = 50.0  # GB/s: one NDR rail, 400 Gb/s, per direction
CLOSE = 0.90  # >= 90% of line rate counts as "close to max"

outs = sorted(glob.glob(os.path.join(D, "osu-*.out")), key=os.path.getmtime)
if not outs:
    raise SystemExit("no output")
f = outs[-1]
txt = open(f, errors="replace").read()
head = txt.split("===== STACK")[0]
nodes = re.findall(r"^node: (\S+) others=(\d+)", head, re.M)
date = (re.search(r"^date: (.*)", head, re.M) or [None, "?"])[1]
job = (re.search(r"^job: (\S+)", head, re.M) or [None, "?"])[1]

def table(block):
    rows = {}
    for m in re.finditer(r"^\s*(\d+)\s+([\d.]+)\s*$", block, re.M):
        rows[int(m.group(1))] = float(m.group(2))
    return rows

res = []
for blk in txt.split("===== STACK")[1:]:
    name = blk.split()[0]
    envv = (re.search(r"env=(\S+)", blk) or [None, "-"])[1]
    envv = "default" if envv == "-" else envv
    mp = re.search(r"modulepath=(\S+)", blk).group(1)
    ver = (re.search(r"^mpirun \(Open MPI\) (\S+)", blk, re.M) or [None, "?"])[1]
    pre = re.findall(r"^preflight (\S+): (.*)$", blk, re.M)
    r = {"name": name, "env": envv, "mp": mp, "ver": ver, "pre": pre, "tests": {}}
    for t in ("osu_latency", "osu_bw", "osu_bibw"):
        m = re.search(r"----- TEST %s\n(.*?)----- RC %s (\d+)" % (t, t), blk, re.S)
        if not m:
            r["tests"][t] = {"rc": None, "rows": {}, "err": "not run"}
            continue
        body, rc = m.group(1), int(m.group(2))
        err = ""
        if rc != 0:
            ls = [l.strip() for l in body.splitlines()]
            e = [l for l in ls if re.search(r"Fatal:|PMIX ERROR|cannot open shared object|Error message:", l)] or \
                [l for l in ls if re.search(r"error|fatal|abort|timeout|not found", l, re.I)]
            err = (e[0] if e else "exit code %d" % rc)[:300]
            if rc == 124:
                err = "killed by 300 s timeout; " + err
        r["tests"][t] = {"rc": rc, "rows": table(body), "err": err}
    res.append(r)

L = []
L.append("# %s\n" % TITLE)
L.append("- Generated: %s by `analyze-osu-ompi508.py` (job `%s`)" % (time.strftime("%F %T"), JOBS.get(SUB, "?")))
L.append("- Job %s, run %s, partition `mit_testing`, no QoS; nodes picked by Slurm: %s" % (
    job, date, ", ".join("%s (other jobs on node: %s)" % n for n in nodes)))
L.append("- 1 rank (1 core) per node, `mpirun -np 2 --map-by ppr:1:node`, host memory; settings per row (default = no UCX/MCA tuning)")
L.append("- OSU Micro-Benchmarks 7.5-1 built against each module (%s)" % ", ".join(sorted({"`../mpi-p2p/install/%s`" % r for r in re.findall(r"build=(\S+)", txt)})))
L.append("- Max bandwidth: one MPI pair uses one NDR rail, 400 Gb/s = **%.0f GB/s per direction** (%.0f GB/s bidirectional). \"Close to max\" = at least %d%% of that.\n" % (LINE, 2 * LINE, CLOSE * 100))

L.append("## Verdict\n")
summ = []
for r in res:
    t = r["tests"]
    lat = t["osu_latency"]["rows"]; bw = t["osu_bw"]["rows"]; bi = t["osu_bibw"]["rows"]
    ok = all(t[x]["rc"] == 0 and t[x]["rows"] for x in t)
    pbw = max(bw.values()) / 1000 if bw else None  # MB/s -> GB/s
    pbi = max(bi.values()) / 1000 if bi else None
    l8 = lat.get(8)
    r.update(ok=ok, pbw=pbw, pbi=pbi, l8=l8)
    label = "%s%s, %s" % (LABEL.get(r["name"], r["name"]), "" if r["mp"] == "-" else " module (`%s`)" % r["mp"], "default settings" if r["env"] == "default" else "with `%s`" % r["env"])
    if not ok and pbw is None and l8 is None:
        v = "**Does not work.** No test completed."
    elif not ok:
        v = "**Partly works.** Some tests failed."
    elif pbw / LINE >= CLOSE:
        v = "**Works, close to max bandwidth**: osu_bw peak %.1f GB/s = %.0f%% of %.0f GB/s." % (pbw, 100 * pbw / LINE, LINE)
    else:
        v = "**Works, but NOT close to max bandwidth**: osu_bw peak %.1f GB/s = only %.0f%% of %.0f GB/s." % (pbw, 100 * pbw / LINE, LINE)
    L.append("- %s: %s" % (label, v))
L.append("")

L.append("## Results\n")
L.append("| Module | Settings | Open MPI | osu_latency 8 B (µs) | osu_bw peak (GB/s) | % of 50 GB/s | osu_bibw peak (GB/s) | % of 100 GB/s | All tests OK |")
L.append("|---|---|---|---:|---:|---:|---:|---:|---|")
fmt = lambda v, p="%.2f": (p % v) if v is not None else "—"
for r in res:
    L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        LABEL.get(r["name"], r["name"]) if r["mp"] == "-" else r["mp"], r["env"], r["ver"], fmt(r["l8"]), fmt(r["pbw"], "%.1f"),
        fmt(r["pbw"] and 100 * r["pbw"] / LINE, "%.0f%%"),
        fmt(r["pbi"], "%.1f"), fmt(r["pbi"] and 100 * r["pbi"] / (2 * LINE), "%.0f%%"),
        "yes" if r["ok"] else "no"))
L.append("\nOSU reports bandwidth in MB/s (10^6 B/s); shown here as GB/s.\n")

L.append("## Failures and preflight\n")
for r in res:
    for n, p in r["pre"]:
        L.append("- `%s` (%s) preflight %s: %s" % (r["name"], r["env"], n, "libraries OK" if p == "OK" else "missing libs: " + p))
    for x, t in r["tests"].items():
        if t["rc"] != 0:
            L.append("- `%s` (%s) %s: %s" % (r["name"], r["env"], x, t["err"]))
L.append("")

if SUB == "nvhpc-26.1":
    gpus = re.findall(r"^gpu (\S+): (.*)$", head, re.M)
    L.append("""## Context

CUDA-aware MPI test: 1 core + 1 B200 per node (Slurm picks the GPU). `D D`
runs use `osu_* -d cuda D D` (send and receive buffers in GPU memory, data moved
by UCX with GPUDirect RDMA). With default settings UCX may also use the
node-local HCAs mlx5_0-3 and abort with a UD timeout (see
`../ompi-5.0.8/report.md`). The "8 NDR HCAs" row lets UCX choose among
mlx5_4/7/8/9/10/13/14/15, so it can use the HCA nearest the GPU; the
`mlx5_4:1` row may cross PCIe/CPU sockets if the GPU is far from mlx5_4.

GPUs given by Slurm: %s
""" % "; ".join("%s %s" % g for g in gpus))
if SUB == "ucx-net-devices":
    L.append("""## Context

With default settings, UCX on B200 also uses the node-local HCAs mlx5_0-3
(100 Gb/s, own subnet manager, same `fe80::` subnet prefix as the NDR fabric),
which cannot reach other nodes, and MPI aborts with `UD endpoint ... unhandled
timeout error` (see `../ompi-5.0.8/report.md`). This test restricts UCX to the
NDR HCA `mlx5_4` with `UCX_NET_DEVICES=mlx5_4:1` in the job script, using the
same `openmpi/4.1.4` module and OSU build; the default row is the control on
the same node pair.
""")
if SUB == "ompi-5.0.8": L.append("""## Why MPI fails on B200 but not on mit_normal / L40S nodes

Each B200 node has two separate InfiniBand networks, and both use the same
default subnet prefix `fe80::`:

| HCAs | Rate | Network |
|---|---|---|
| mlx5_4/7/8/9/10/13/14/15 | 400 Gb/s NDR | cluster fabric (SM LID 0x719), node-to-node traffic |
| mlx5_0/1/2/3 | 100 Gb/s (2X HDR) | node-local network (own SM, LID 0x1 on every node checked), cannot reach other nodes |

(mlx5_5/6/11/12 are DOWN.) mit_normal / L40S nodes have one HCA on the
cluster fabric (same SM LID 0x719), so there is nothing to mix up.

UCX decides that two HCAs can reach each other by comparing subnet prefixes.
Because the node-local HCAs mlx5_0-3 also use `fe80::`, UCX treats them as
reachable from the other node and uses them for its UD connection setup. The
packets never reach the other node, and the rank aborts with `ud_ep.c: Fatal:
UD endpoint ... unhandled timeout error`. This happens with UCX 1.16
(`openmpi/4.1.4`) and UCX 1.19 (Spack `openmpi/5.0.8`). Raw `ib_write_bw`
works because it is pinned to `mlx5_4`, and NCCL works because it is given
the 8 NDR HCAs explicitly.

Suggested fix, to be confirmed with one test: limit UCX to the NDR HCAs, e.g.
`UCX_NET_DEVICES=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1`
(or only `mlx5_4:1` for a one-rail p2p test). A site-wide option for ORCD
admins is to set this in the openmpi modules on B200 nodes, or to give the
node-local network its own subnet prefix.

Other module-specific problems:

- Spack `openmpi/5.0.8`, default settings: fails earlier, in `MPI_Init`, with
  a PMIx `gds/shmem2` shared-memory attach error. `PMIX_MCA_gds=hash` fixes
  that step, and then the UCX timeout above happens.
- core `openmpi/5.0.8` (`/orcd/software/core/001/modulefiles`): cannot start
  on B200 because it links `libhcoll.so.1` / `libocoms.so.0`
  (`/opt/mellanox/hcoll`), which the B200 nodes do not have.
""")
L.append("## Bandwidth by message size (GB/s)\n")
sizes = sorted({s for r in res for x in ("osu_bw", "osu_bibw") for s in r["tests"][x]["rows"] if s >= 65536})
if sizes:
    hdr = " | ".join("%s/%s bw | %s/%s bibw" % (r["name"], r["env"], r["name"], r["env"]) for r in res)
    L.append("| Size (B) | %s |" % hdr)
    L.append("|---:|" + "---:|" * (2 * len(res)))
    for s in sizes:
        cells = []
        for r in res:
            for x in ("osu_bw", "osu_bibw"):
                v = r["tests"][x]["rows"].get(s)
                cells.append("%.1f" % (v / 1000) if v is not None else "—")
        L.append("| %d | %s |" % (s, " | ".join(cells)))
L.append("\nRaw output: `%s`" % os.path.basename(f))

out = "\n".join(L) + "\n"
open(os.path.join(D, "report.md"), "w").write(out)
print(out)
