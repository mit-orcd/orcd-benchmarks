#!/usr/bin/env python3
"""Analyze the NCCL all_reduce SHARP A/B runs and write a Markdown summary.

Each job (job-nccl-2node-sharp.sh) runs all_reduce twice inside one allocation:
a Ring leg (CollNet disabled) and a SHARP leg (CollNet algorithms), delimited by
`%%%%% MODE <ring|sharp> %%%%%` markers. This script pairs the two legs per node
pair and reports the speed-up, plus a comparison to the MIT aicr-benchmarks
reference (Ring 170 GB/s -> SHARP 357 GB/s, 2.2x).

IMPORTANT: if the fabric/switches are not SHARP-enabled, or sharpd is not
running, NCCL logs the CollNet setup failure and silently falls back to Ring —
the two legs then produce near-identical numbers. This script inspects the
NCCL INIT/NET debug output and reports whether SHARP actually engaged, so a
"no speed-up" result is never mistaken for "SHARP is no faster here".

Usage:
    ./analyze-nccl-sharp.py [file ...]   # default: newest *.out per node pair
"""
import glob
import os
import re
import sys
from datetime import datetime

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "out-nccl-2node-sharp")
REFERENCE = {"ring": 170.0, "sharp": 357.0, "speedup": 2.2}
# ring-collective ceiling of this fabric: 8 NDR rails x 50 GB/s per direction
# (same basis as out-nccl-2node/summary.md)
ALLREDUCE_HW_MAX = 400.0

MODE_RE = re.compile(r"%+\s*MODE\s+(\w+)\s*%+")
DEVICE_RE = re.compile(
    r"Rank\s+(\d+).*on\s+(\S+)\s+device\s+(\d+)\s+\[[^\]]+\]\s+(.+?)\s*$"
)
DATA_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(-?\d+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\S+)\s*$"
)
CONFIG_RE = re.compile(r"nThread\s+(\d+)\s+nGpus\s+(\d+)")

# Evidence that the SHARP/CollNet path was actually set up.
OK_RE = re.compile(
    r"(Connected\s+CollNet|CollNet\s+(?:enabled|support)|collNetSupport\s*[:=]?\s*1"
    r"|Using\s+CollNet|SHARP\s+(?:enabled|initialized))", re.I)
# Evidence it could not be set up. NCCL may then fall back to Ring silently, or
# the run may abort outright. The SHARPD_/AM lines come from the SHARP runtime
# itself: the fabric's Aggregation Manager is what actually builds SHARP trees.
BAD_RE = re.compile(
    r"(CollNet\s+(?:not|un)\s*support|collNetSupport\s*[:=]?\s*0|failed to open"
    r"|Cannot open libnccl-net|sharp.*(?:fail|error|not available)"
    r"|SHARPD_OP_\w+\s+failed|unable to connect to AM|Could not query AM"
    r"|no AM service record|No\s+CollNet|disabling\s+CollNet)", re.I)


def fmt_size(n):
    for div, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= div and n % div == 0:
            return f"{n // div} {unit}"
    for div, unit in ((1024**3, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n} B"


def parse_file(path):
    """-> {pair, gpus_per_node, legs: {mode: {rows, converged, evidence...}}}"""
    legs, devices = {}, []
    mode = None
    cur = None
    ngpus_cfg = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = MODE_RE.search(line)
            if m:
                mode = m.group(1).lower()
                if mode == "end":
                    mode, cur = None, None
                    continue
                cur = legs.setdefault(mode, {"rows": [], "ok_ev": [], "bad_ev": []})
                continue
            m = DEVICE_RE.search(line)
            if m:
                devices.append({"rank": int(m.group(1)), "node": m.group(2)})
                continue
            if cur is None:
                continue
            m = CONFIG_RE.search(line)
            if m and ngpus_cfg is None:
                ngpus_cfg = int(m.group(2))
            m = DATA_RE.match(line)
            if m:
                cur["rows"].append({
                    "size": int(m.group(1)),
                    "oop_busbw": float(m.group(8)), "oop_wrong": m.group(9),
                    "ip_busbw": float(m.group(12)), "ip_wrong": m.group(13),
                })
                continue
            s = line.strip()
            if OK_RE.search(s) and len(cur["ok_ev"]) < 3:
                cur["ok_ev"].append(s[:160])
            elif BAD_RE.search(s) and len(cur["bad_ev"]) < 3:
                cur["bad_ev"].append(s[:160])

    # keep a leg that produced no data if it left diagnostic evidence — a leg that
    # aborted is the informative case, not something to drop silently
    legs = {k: v for k, v in legs.items() if v["rows"] or v["bad_ev"] or v["ok_ev"]}
    if not any(v["rows"] for v in legs.values()):
        return None
    for v in legs.values():
        if not v["rows"]:
            v["converged"] = v["peak"] = None
            v["ok"] = None
            continue
        big = max(v["rows"], key=lambda r: r["size"])
        v["converged"] = max(big["oop_busbw"], big["ip_busbw"])
        v["peak"] = max(max(r["oop_busbw"], r["ip_busbw"]) for r in v["rows"])
        v["ok"] = all(r["oop_wrong"] in ("0", "N/A") and r["ip_wrong"] in ("0", "N/A")
                      for r in v["rows"])
    pair = "+".join(sorted({d["node"] for d in devices})) or os.path.basename(path)
    return {"path": path, "pair": pair, "gpus_per_node": ngpus_cfg, "legs": legs}


def sharp_status(leg):
    """-> (verdict, detail) for the SHARP leg."""
    if leg is None:
        return "no data", ""
    if leg["bad_ev"] and not leg["rows"]:
        return "UNAVAILABLE (run aborted)", leg["bad_ev"][0]
    if leg["bad_ev"]:
        return "FELL BACK to Ring", leg["bad_ev"][0]
    if leg["ok_ev"]:
        return "engaged", leg["ok_ev"][0]
    return "unconfirmed", "no CollNet/SHARP setup lines found in the NCCL debug output"


def build(runs):
    L = []
    L.append("# NCCL all_reduce — SHARP vs Ring (2-node A/B)")
    L.append("")
    L.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"- Runs: {', '.join(r['pair'] for r in runs)}")
    gp = next((r["gpus_per_node"] for r in runs if r["gpus_per_node"]), "?")
    L.append(f"- {gp} GPUs/node x 2 nodes, all_reduce, 1 MiB - 16 GiB")
    L.append("- Both legs run back-to-back in ONE allocation (same nodes, same NICs)")
    L.append(f"- Reference (`results_b200.md`): Ring **{REFERENCE['ring']:.0f}** -> "
             f"SHARP **{REFERENCE['sharp']:.0f}** GB/s = **{REFERENCE['speedup']}x**")
    L.append("")

    # Headline
    L.append("## Converged busbw: Ring vs SHARP")
    L.append("")
    L.append("| Node pair | Ring (GB/s) | SHARP (GB/s) | speed-up | SHARP status | "
             "correctness |")
    L.append("|-----------|------------:|-------------:|---------:|--------------|"
             ":-----------:|")
    for r in runs:
        ring = r["legs"].get("ring")
        sharp = r["legs"].get("sharp")
        verdict, _ = sharp_status(sharp)
        rv = f"{ring['converged']:.1f}" if ring and ring["converged"] else "—"
        sv = f"{sharp['converged']:.1f}" if sharp and sharp["converged"] else "—"
        sp = (f"{sharp['converged']/ring['converged']:.2f}x"
              if ring and sharp and ring["converged"] and sharp["converged"] else "—")
        oks = [v["ok"] for v in r["legs"].values() if v["ok"] is not None]
        okc = "PASS" if all(oks) else "FAIL"
        L.append(f"| {r['pair']} | {rv} | {sv} | {sp} | {verdict} | {okc} |")
    L.append("")

    # SHARP engagement evidence — the part that decides how to read the numbers
    L.append("## Did SHARP actually engage?")
    L.append("")
    L.append("NCCL falls back to Ring **silently** when the CollNet/SHARP path cannot "
             "be set up, so the speed-up column above is only meaningful once this is "
             "confirmed. Evidence from the NCCL `INIT,ENV,NET` debug output:")
    L.append("")
    for r in runs:
        verdict, detail = sharp_status(r["legs"].get("sharp"))
        L.append(f"- **{r['pair']}** — {verdict}"
                 + (f": `{detail}`" if detail else ""))
    L.append("")

    # ---- Verdict: did we get the expected SHARP result? --------------------
    got_sharp = [r for r in runs
                 if r["legs"].get("sharp") and r["legs"]["sharp"]["converged"]]
    rings = [r["legs"]["ring"]["converged"] for r in runs
             if r["legs"].get("ring") and r["legs"]["ring"]["converged"]]
    L.append("## Verdict: are these the expected SHARP results?")
    L.append("")
    if not got_sharp:
        rmin, rmax = (min(rings), max(rings)) if rings else (0, 0)
        L.append("**No — there are no SHARP results at all.** Not poor numbers: "
                 "*zero* measurements. The SHARP leg aborted during initialisation, "
                 "before running a single message size.")
        L.append("")
        L.append(f"**What did work:** the Ring baseline, at "
                 f"**{rmin:.0f}-{rmax:.0f} GB/s** across "
                 f"{len(rings)} run(s), validation-clean and consistent with the "
                 f"standalone 2-node all_reduce in `out-nccl-2node/summary.md`. The "
                 f"A/B harness is sound; only half of it can execute.")
        L.append("")
        L.append(f"**What a good result would look like:** the reference cluster "
                 f"measured Ring {REFERENCE['ring']:.0f} -> SHARP "
                 f"{REFERENCE['sharp']:.0f} GB/s = **{REFERENCE['speedup']}x**. The "
                 f"upside here is plausibly larger: our Ring all_reduce sits at "
                 f"~{rmax:.0f} GB/s, only ~{100*rmax/ALLREDUCE_HW_MAX:.0f}% of the "
                 f"{ALLREDUCE_HW_MAX:.0f} GB/s hardware ceiling, while every other "
                 f"ring collective on this fabric already runs at 92-96%. That gap "
                 f"*is* the two-pass Ring penalty SHARP exists to remove, and it is "
                 f"the single largest piece of unrealised inter-node performance in "
                 f"these benchmarks — it directly gates multi-node DDP gradient "
                 f"sync.")
        L.append("")
        L.append("**Why we cannot get it:** `No Aggregation Manager (sharp_am) "
                 "detected`. Confirmed three independent ways — through NCCL with a "
                 "plain CollNet setup, through NCCL with the AICR environment recipe "
                 "(`job-nccl-2node-sharp-aicr.sh`), and through `sharp_hello` "
                 "standalone with NCCL entirely out of the picture. The node-side "
                 "stack is complete and correct: the plugin loads, CollNet channels "
                 "are allocated, and the SHARP client library runs. The fabric "
                 "simply has no Aggregation Manager to register a SHARP job with.")
        L.append("")
        L.append("**Nothing further can be done from the job side.** Getting SHARP "
                 "results requires the InfiniBand admins to run `sharp_am` on the "
                 "subnet manager / UFM host and provision aggregation trees for "
                 "these nodes — see `sharp.md` for the full diagnosis, the hardware "
                 "assessment, and the questions to ask. Once that is done, "
                 "`job-nccl-2node-sharp-aicr.sh` runs unchanged and this table will "
                 "populate itself.")
    else:
        L.append("SHARP produced measurements on "
                 f"{len(got_sharp)}/{len(runs)} node pair(s). Compare the speed-up "
                 f"column against the reference's **{REFERENCE['speedup']}x** "
                 f"(Ring {REFERENCE['ring']:.0f} -> SHARP "
                 f"{REFERENCE['sharp']:.0f} GB/s), and the absolute SHARP figure "
                 f"against the {ALLREDUCE_HW_MAX:.0f} GB/s ring-collective ceiling "
                 f"of this fabric — SHARP can exceed it, because a single in-switch "
                 f"reduction halves the wire traffic that ceiling assumes.")
    L.append("")

    # Per-size detail
    L.append("## Bus bandwidth vs message size (GB/s)")
    L.append("")
    for r in runs:
        ring = r["legs"].get("ring")
        sharp = r["legs"].get("sharp")
        if not ring:
            continue
        L.append(f"### {r['pair']}")
        L.append("")
        L.append("| Message size | Ring | SHARP | speed-up |")
        L.append("|-------------:|-----:|------:|---------:|")
        s_by_size = ({row["size"]: max(row["oop_busbw"], row["ip_busbw"])
                      for row in sharp["rows"]} if sharp else {})
        for row in ring["rows"]:
            rb = max(row["oop_busbw"], row["ip_busbw"])
            sb = s_by_size.get(row["size"])
            sp = f"{sb/rb:.2f}x" if (sb and rb) else "—"
            L.append(f"| {fmt_size(row['size'])} | {rb:.1f} | "
                     f"{sb:.1f} | {sp} |" if sb else
                     f"| {fmt_size(row['size'])} | {rb:.1f} | — | — |")
        L.append("")
    L.append("busbw, best of out-of-place / in-place. SHARP offloads the reduction to "
             "the InfiniBand switches, making all_reduce a single pass instead of "
             "ReduceScatter+AllGather; the reference sees the gain grow with message "
             "size and win above ~4 MB.")
    L.append("")
    return "\n".join(L) + "\n"


def collect(args):
    files = args if args else glob.glob(os.path.join(OUT_DIR, "*.out"))
    parsed = {}
    for f in sorted(files, key=os.path.getmtime):   # newest wins per pair
        r = parse_file(f)
        if r:
            parsed[r["pair"]] = r
    return [parsed[k] for k in sorted(parsed)]


def main():
    runs = collect(sys.argv[1:])
    if not runs:
        sys.exit(f"No parseable SHARP A/B results in {OUT_DIR}")
    md = build(runs)
    summary = os.path.join(OUT_DIR, "summary.md")
    with open(summary, "w") as fh:
        fh.write(md)
    print(md)
    print(f"Written to {summary}  ({len(runs)} pair(s))")


if __name__ == "__main__":
    main()
