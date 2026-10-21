#!/usr/bin/env python3
"""Result of atom/run_kimi_atom_mtp.sh (Kimi-K3, 8 x MI355X, ATOM + DSpark MTP).

  analyze_atom_mtp.py <logs/<node>/kimi-atom-mtp/atom/kimi_atom_mtp_TS>

Writes
  results/ubuntu/kimi-atom-mtp.md            all points
  results/mi355x-vs-b200-kimi-k3.md          adds "Ours MI355X ATOM +MTP" (other columns stay)
  results/status.txt                         one line
Env MTP_MD / MTP_STATUS / MTP_SUM override the paths (tests).
"""
import glob, json, os, re, statistics, sys

R = "/orcd/data/orcd/022/benchmarks/amd-ubuntu/results"
MD = os.environ.get("MTP_MD", f"{R}/mi355x-vs-b200-kimi-k3.md")
STATUS = os.environ.get("MTP_STATUS", f"{R}/status.txt")
SUMMD = os.environ.get("MTP_SUM", f"{R}/ubuntu/kimi-atom-mtp.md")
# our ATOM baseline without MTP (results/node6100/kimi-cloud/atom.csv, sweep_20261002_025708)
NOSPEC_CSV = f"{R}/node6100/kimi-cloud/atom.csv"
COL = "**Ours** MI355X ATOM +MTP"
NSCOL = "**Ours** MI355X ATOM (no MTP)"   # K=0 arm, same image; goes right before COL
f1 = lambda x: f"{x:,.1f}"


def nospec():
    d = {}
    for l in open(NOSPEC_CSV).read().splitlines()[1:]:
        p = l.split(",")
        if p[0] == "Kimi-K3":
            d[int(p[2])] = (float(p[3]), 1000 / float(p[8]))
    return d


def rep(s, old, new, what):
    if old not in s:
        print("WARN: text not found for", what)
        return s
    return s.replace(old, new, 1)


def add_col(s, header_start, nth, new_head, vals, sep="---:", before_last=False):
    """append a column to the nth table whose header starts with header_start
    (before_last: insert it before the last column instead, i.e. right before its InferenceX twin)"""
    lines = s.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(header_start)]
    if len(idx) <= nth:
        print("WARN: table not found:", header_start, nth)
        return s
    i = idx[nth]
    def app(line, cell):
        if before_last:
            p = line.rstrip().split("|")   # ['', c1, ..., cN, '']
            p.insert(len(p) - 2, f" {cell} ")
            return "|".join(p)
        return line.rstrip().rstrip("|") + f"| {cell} |"
    lines[i] = app(lines[i], new_head)
    lines[i + 1] = app(lines[i + 1], sep)
    k = i + 2
    while k < len(lines) and lines[k].startswith("|"):
        c = lines[k].split("|")[1].strip()
        c = int(c) if c.isdigit() else c
        lines[k] = app(lines[k], vals.get(c, "—"))
        k += 1
    return "\n".join(lines)


def turn(p1, p8, p, a, w):
    t1, u1, t8, u8 = p1["ttft"] / 1000, p1["pu"], p8["ttft"] / 1000, p8["pu"]
    P, A, W = t1 + (p - 1) / u1, t8 + (a - 1) / u8, t1 + (w - 1) / u1
    return P, A, W, P + A + W, P + A + t1, u1 * 0.75


def main():
    run = sys.argv[1].rstrip("/")
    pts, kof, acc, err, k0 = {}, {}, {}, "", {}
    for d in sorted(glob.glob(f"{run}/k*")):
        k = int(os.path.basename(d)[1:])
        log = f"{d}/atom_server.log"
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        a = [float(x) for x in re.findall(r"(?:[Aa]ccept(?:ance)?[ _]len(?:gth)?)[^0-9]{0,20}([\d.]+)", txt)]
        if a:
            acc[k] = statistics.median(a)
        if os.path.exists(f"{d}/FAILED"):
            e = [l.strip() for l in txt.splitlines() if re.search(r"(Error|Exception):", l)]
            err = err or (e[-1][-300:] if e else "see server log")
        for j in glob.glob(f"{d}/c*.json"):
            r = json.load(open(j))
            c = int(re.search(r"c(\d+)\.json$", j).group(1))
            if r.get("completed", 0) > 0 and r.get("median_tpot_ms") and k == 0:
                k0[c] = (r["output_throughput"], 1000 / r["median_tpot_ms"], r["median_ttft_ms"])
            elif r.get("completed", 0) > 0 and r.get("median_tpot_ms"):
                pts[c] = dict(tps=r["output_throughput"], pu=1000 / r["median_tpot_ms"], tpot=r["median_tpot_ms"],
                              ttft=r["median_ttft_ms"])
                kof[c] = k
    # no-MTP reference: the K=0 arm of this run (same image) where it ran, else our ATOM baseline
    ns = k0 or nospec()
    nsref = "the no-MTP arm of this run (same image)" if k0 else "our ATOM baseline (`node6100/kimi-cloud/atom.csv`)"
    imgf = f"{run}/images.txt"
    img = open(imgf).read().split(" ->")[0].strip() if os.path.exists(imgf) else "rocm/atom-dev"
    s = open(MD).read()
    klab = " at 1 user, ".join(f"{k}" for k in sorted(set(kof.values()), reverse=True)) + (" above" if len(set(kof.values())) > 1 else "")

    # summary file
    D = ["# Kimi-K3 on 8 × MI355X: ATOM + DSpark MTP", "",
         f"Run: `{run}`. Image `{img}`, our ATOM baseline flags (TP8, max-num-seqs 64, "
         "no prefix caching), ISL/OSL 1024/1024, `--ignore-eos`, 10 × users prompts; `--method dspark --draft-model "
         "Inferact/Kimi-K3-DSpark`, real acceptance (no forced acceptance length). TPOT = time per output token, "
         f"TTFT = time to first token. No MTP = {nsref}.", ""]
    if acc:
        D += ["Acceptance length (median of the server's log lines): " + "; ".join(f"{v:.2f} at draft length {k}" for k, v in acc.items()) + ".", ""]
    D += ["| Users | draft length | out tok/s | per-user tok/s | TPOT med (ms) | TTFT med (ms) | per-user, no MTP | MTP / no MTP |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(pts):
        p, n = pts[c], ns.get(c)
        D.append(f"| {c} | {kof[c]} | {f1(p['tps'])} | {f1(p['pu'])} | {p['tpot']:.2f} | {p['ttft']:.0f} | "
                 + (f"{f1(n[1])} | {p['pu'] / n[1]:.2f}x |" if n else "– | – |"))
    if err:
        D += ["", f"Server error: `{err}`"]
    open(SUMMD, "w").write("\n".join(D) + "\n")

    forced = ("| Acceptance length (ATOM +MTP) | **MI355X** | {ours} | **forced**: `--spec-decode-acceptance-length` 3.75 at 1 user, "
              "2.51 above (*\"the rejection sampler ignores the real draft/target agreement and force-accepts draft tokens\"*) | "
              "the InferenceX ATOM +MTP numbers do not come from real acceptance |")
    anchor = next((l for l in s.split("\n") if l.startswith("| Serving engine | **MI355X** |")), None)

    if not pts:
        if anchor:
            s = rep(s, anchor, anchor + "\n" + forced.format(ours="– (our ATOM + MTP run did not start)"), "§2 forced")
        s = rep(s, "**On MI355X it is on in our vLLM recipe 1** (up to 14 users).",
                f"**On MI355X it is on in our vLLM recipe 1** (up to 14 users); our ATOM + MTP run did not start (`{err[:200]}`).", "§6 fail")
        open(MD, "w").write(s)
        line = f"MI355X ATOM + MTP: no point completed ({run})"
        open(STATUS, "a").write(line + "\n"); print(line)
        return

    s = rep(s, next((l for l in s.split("\n") if l.startswith("- **8 × MI355X, vLLM recipe 1**")), "@@"),
            next((l for l in s.split("\n") if l.startswith("- **8 × MI355X, vLLM recipe 1**")), "")
            + f"\n- **8 × MI355X, ATOM + MTP** (one node, TP8): our ATOM flags (max-num-seqs 64), image `{img}` "
              f"plus DSpark MTP (`--method dspark`, `Inferact/Kimi-K3-DSpark`), draft length {klab}, as in the InferenceX ATOM recipe (our ATOM baseline image, `nightly_202608111555`, crashes with this draft). "
              "Detail: `ubuntu/kimi-atom-mtp.md`.", "intro")
    if anchor:
        s = rep(s, anchor, anchor + "\n" + forced.format(ours="real (measured draft/target agreement)"), "§2 forced")

    head = "| Conc | **Ours** B200 TP8×PP2 (no MTP) |"
    if k0:
        s = add_col(s, head, 0, NSCOL, {c: f1(v[0]) for c, v in k0.items()}, before_last=True)
        s = add_col(s, head, 1, NSCOL, {c: f1(v[1]) for c, v in k0.items()}, before_last=True)
    s = add_col(s, head, 0, COL, {c: f1(p["tps"]) for c, p in pts.items()}, before_last=True)
    s = add_col(s, head, 1, COL, {c: f1(p["pu"]) for c, p in pts.items()}, before_last=True)
    if 64 in pts:
        p = pts[64]
        s = rep(s, "| **InferenceX** B200 +MTP, agentic | 32 |",
                f"| **Ours** MI355X ATOM +MTP, 1024/1024 | 64 | {f1(p['tps'])} | {p['tps'] / 64:.1f} | {f1(p['pu'])} | **{p['tps'] / 64 / p['pu']:.2f}** |\n"
                "| **InferenceX** B200 +MTP, agentic | 32 |", "duty row")

    # next numbered point in §4
    lines = s.split("\n")
    h = next((i for i, l in enumerate(lines) if l.startswith("**What the table shows, in ")), None)
    if h is not None and 1 in pts and 1 in ns:
        j = h + 1
        while j < len(lines) and not re.match(r"\d+\. ", lines[j]):
            j += 1
        while j + 1 < len(lines) and re.match(r"\d+\. ", lines[j + 1]):
            j += 1
        n = int(re.match(r"(\d+)\.", lines[j]).group(1)) + 1 if re.match(r"\d+\.", lines[j]) else 1
        g = pts[1]["pu"] / ns[1][1]
        lines.insert(j + 1, f"{n}. **MTP is worth {g:.1f}× at c=1 on our MI355X ATOM** ({f1(pts[1]['pu'])} vs {f1(ns[1][1])} without MTP, "
                            f"real acceptance). InferenceX's MI355X ATOM +MTP gets 127.2 at c=1, but with a forced acceptance length (§2).")
        words = {3: "three", 4: "four", 5: "five", 6: "six"}
        lines[h] = re.sub(r"in \w+ points", f"in {words.get(n, n)} points", lines[h])
        s = "\n".join(lines)

    if 1 in pts and 8 in pts:
        rows = []
        anchors = [i for i, l in enumerate(s.split("\n")) if l.startswith("| **Ours** MI355X vLLM recipe 1 (+MTP) |")]
        for idx, (p, a, w) in zip(reversed(anchors), ((30, 70, 100), (200, 500, 800))):
            P, A, W, T, F, Rr = turn(pts[1], pts[8], p, a, w)
            L = s.split("\n")
            if 1 in k0 and 8 in k0:   # the no-MTP arm, same image, right before the +MTP row
                q = lambda v: dict(ttft=v[2], pu=v[1])
                nP, nA, nW, nT, nF, nR = turn(q(k0[1]), q(k0[8]), p, a, w)
                L.insert(idx + 1, f"| **Ours** MI355X ATOM (no MTP) | {nP:.1f} s | {nA:.1f} s | {nW:.1f} s | **{nT:.1f} s** | {nF:.1f} s | {nR:.0f} words/s |")
                idx += 1
                rows.append(nR)
            L.insert(idx + 1, f"| **Ours** MI355X ATOM +MTP | {P:.1f} s | {A:.1f} s | {W:.1f} s | **{T:.1f} s** | {F:.1f} s | {Rr:.0f} words/s |")
            s = "\n".join(L)
            rows.append(Rr)
        m = re.search(r"Every row prints at (\d+)–(\d+) words/s", s)
        if m:
            lo, hi = min(int(m.group(1)), *rows), max(int(m.group(2)), *rows)
            s = s.replace(m.group(0), f"Every row prints at {lo:.0f}–{hi:.0f} words/s", 1)

    g = [(c, pts[c]["pu"], ns[c][1]) for c in (1, 8, 32) if c in pts and c in ns]
    if g:
        tab = ["", "Measured gain in our MI355X ATOM data (MTP vs no MTP, same hardware, 1024/1024, real acceptance):", "",
               "| | " + " | ".join(f"c={c}" for c, _, _ in g) + " |", "|---|" + "---:|" * len(g),
               "| MI355X ATOM per-user tok/s, **no MTP** | " + " | ".join(f1(n) for _, _, n in g) + " |",
               "| MI355X ATOM per-user tok/s, **MTP** | " + " | ".join(f1(m) for _, m, _ in g) + " |",
               "| ratio | " + " | ".join(f"**{m / n:.1f}×**" for _, m, n in g) + " |", ""]
        s = rep(s, "\nCeiling is draft length × acceptance rate", "\n".join(tab) + "\nCeiling is draft length × acceptance rate", "§6 gain")
    if acc:
        line = next((l for l in s.split("\n") if l.startswith("Ceiling is draft length")), None)
        if line:
            s = s.replace(line, line + " With our ATOM + MTP it is "
                          + "; ".join(f"{v:.2f} at draft length {k}" for k, v in acc.items()) + " (`ubuntu/kimi-atom-mtp.md`).", 1)

    s = add_col(s, "| Step | MI355X (our vLLM recipe 1) |", 0, "MI355X (our ATOM)", {}, sep="---")
    for old, new in (("| 1. Speculator |", "`Inferact/Kimi-K3-DSpark`"),
                     ("| 2. `pard_token` shim |", "not needed"),
                     ("| 3. Extra server flags |", "`--method dspark --draft-model --num-speculative-tokens`"),
                     ("| 4. Attention backend in the config |", "ATOM default"),
                     ("| 5. Draft length |", f"{klab} (InferenceX ATOM recipe)")):
        line = next((l for l in s.split("\n") if l.startswith(old)), None)
        if line:
            p = line.split("|"); p[-2] = f" {new} "
            s = s.replace(line, "|".join(p), 1)
    s = rep(s, "**On MI355X it is on in our vLLM recipe 1** (up to 14 users).",
            f"**On MI355X it is on in our vLLM recipe 1** (up to 14 users) and in our ATOM + MTP run (draft length {klab}).", "§6 on")
    open(MD, "w").write(s)
    line = (f"MI355X ATOM + MTP: {len(pts)} points; c=1 per-user {f1(pts[1]['pu'])} tok/s ({pts[1]['pu'] / ns[1][1]:.1f}x no MTP)"
            if 1 in pts and 1 in ns else f"MI355X ATOM + MTP: {len(pts)} points") + f"; {MD} updated"
    open(STATUS, "a").write(line + "\n"); print(line)


if __name__ == "__main__":
    main()
