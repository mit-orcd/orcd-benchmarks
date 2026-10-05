#!/usr/bin/env python3
"""Summarize run-rccl-2node.sh output -> results/rccl_2node.{md,csv} (+ busbw plot).

Usage: analyze_rccl_2node.py <rccl_2node_dir> [more dirs...] -o <results_dir>

Parses every <coll>_ppn<P>_np<N>.log (rccl-tests MPI=1 format). Per (collective, PPN):
peak in-place busbw, busbw at a few reference sizes, and small-message latency. Also records
the NIC/transport RCCL picked (from the NCCL_DEBUG=INFO first point) so a run that silently
fell back to TCP sockets is visible in the report rather than just "slow".
"""
import argparse, csv, glob, os, re, sys

ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\w+)\s+(\S+)\s+(-?\d+)\s+(.*)$")
REF_SIZES = [1 << 20, 64 << 20, 1 << 30, 16 << 30]   # 1M, 64M, 1G, 16G


def human(n):
    for unit, s in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if n >= s and n % s == 0:
            return f"{n // s}{unit}"
    return str(n)


def parse(path):
    rows = []
    for line in open(path, errors="replace"):
        m = ROW.match(line)
        if not m:
            continue
        f = m.group(6).split()
        if len(f) < 8:
            continue
        try:  # in-place: time algbw busbw #wrong are the last four fields
            rows.append((int(m.group(1)), float(f[-4]), float(f[-2])))
        except ValueError:
            pass
    return rows


def transport(dirs):
    for d in dirs:
        for log in sorted(glob.glob(os.path.join(d, "*.log"))):
            nets = set()
            for line in open(log, errors="replace"):
                if "NCCL INFO" in line and ("NET/" in line or "Using network" in line):
                    m = re.search(r"Using network (\S+)", line)
                    if m:
                        nets.add(m.group(1))
                    for dev in re.findall(r"ionic_\d+|mlx5_\d+", line):
                        nets.add(dev)
            if nets:
                return ", ".join(sorted(nets))
    return "unknown (no NCCL_DEBUG=INFO lines found)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = []
    for d in a.dirs:
        for log in sorted(glob.glob(os.path.join(d, "*_ppn*_np*.log"))):
            m = re.match(r"(.+)_ppn(\d+)_np(\d+)\.log$", os.path.basename(log))
            if not m:
                continue
            rows = parse(log)
            if not rows:
                recs.append(dict(collective=m.group(1), ppn=int(m.group(2)), ranks=int(m.group(3)),
                                 status="no data", log=log))
                continue
            by = {s: (t, bw) for s, t, bw in rows}
            peak_s, peak = max(((s, bw) for s, _, bw in rows), key=lambda x: x[1])
            r = dict(collective=m.group(1), ppn=int(m.group(2)), ranks=int(m.group(3)), status="ok",
                     min_size=rows[0][0], lat_us=rows[0][1], peak_busbw=peak, peak_size=peak_s,
                     max_size=rows[-1][0], busbw_max_size=rows[-1][2], log=log)
            for s in REF_SIZES:
                r[f"busbw_{human(s)}"] = by.get(s, (None, None))[1]
            recs.append(r)
    if not recs:
        sys.exit("no *_ppn*_np*.log files found")

    cols = ["collective", "ppn", "ranks", "status", "min_size", "lat_us", "peak_busbw", "peak_size",
            "max_size", "busbw_max_size"] + [f"busbw_{human(s)}" for s in REF_SIZES] + ["log"]
    with open(os.path.join(a.out, "rccl_2node.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in recs:
            w.writerow({k: r.get(k, "") for k in cols})

    fmt = lambda v: "-" if v in (None, "") else f"{v:.1f}"
    md = ["# RCCL 2-node (node6100 + node6101, 8x MI355X each)", "",
          f"Source: {', '.join(a.dirs)}", "",
          f"Transport / NICs seen by RCCL: **{transport(a.dirs)}**", "",
          "In-place bus bandwidth (GB/s). `lat` is the in-place time of the smallest message (µs).", "",
          "| collective | PPN | ranks | lat (µs) | 1M | 64M | 1G | 16G | peak busbw | @size |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(recs, key=lambda r: (r["ppn"], r["collective"])):
        if r["status"] != "ok":
            md.append(f"| {r['collective']} | {r['ppn']} | {r['ranks']} | no data | | | | | | |")
            continue
        md.append(f"| {r['collective']} | {r['ppn']} | {r['ranks']} | {fmt(r['lat_us'])} | "
                  + " | ".join(fmt(r[f'busbw_{human(s)}']) for s in REF_SIZES)
                  + f" | **{r['peak_busbw']:.1f}** | {human(r['peak_size'])} |")
    md += ["", "Reference: 8 × 400 Gb/s NICs per node = 400 GB/s/node unidirectional line rate. "
           "For all_reduce, busbw is directly comparable to that per-node link rate.", ""]
    open(os.path.join(a.out, "rccl_2node.md"), "w").write("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in recs:
            if r["status"] != "ok" or r["ppn"] != max(x["ppn"] for x in recs):
                continue
            rows = parse(r["log"])
            ax.plot([s for s, _, _ in rows], [bw for _, _, bw in rows], marker=".", label=r["collective"])
        ax.set_xscale("log", base=2); ax.set_xlabel("message size (bytes)"); ax.set_ylabel("busbw (GB/s)")
        ax.set_title("RCCL 2-node busbw (in-place)"); ax.grid(alpha=.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "rccl_2node_busbw.png"), dpi=120)
    except Exception as e:  # plot is optional
        print("plot skipped:", e)
    print("wrote", os.path.join(a.out, "rccl_2node.md"))


if __name__ == "__main__":
    main()
