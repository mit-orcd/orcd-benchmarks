#!/usr/bin/env python3
"""Result of job-kimi-mtp-b200.sh (Kimi-K3, 2 x 8 B200, TP8 x PP2, DSpark MTP).

  analyze-mtp-b200.py <logs/mtp_TS> <slurm job id>

Writes
  results/kimi-k3-mtp-b200.md                              all arms, all points
  ../amd-ubuntu/results/mi355x-vs-b200-kimi-k3.md          adds the "Ours B200 +MTP" results
                                                           (the B200 no-spec results stay)
  ../amd-ubuntu/results/status.txt                         one line
Env MTP_MD overrides the comparison md path (tests).
"""
import glob, json, os, re, statistics, sys

BENCH = "/orcd/data/orcd/022/benchmarks"
RES = f"{BENCH}/b200-kimi/results"
MD = os.environ.get("MTP_MD", f"{BENCH}/amd-ubuntu/results/mi355x-vs-b200-kimi-k3.md")
STATUS = os.environ.get("MTP_STATUS", f"{BENCH}/amd-ubuntu/results/status.txt")
ARMS = [("mtp7", 7, "TOKENSPEED_MLA"), ("mtp7d", 7, "default"), ("mtp3", 3, None)]
# our B200 no-spec run (job 20916742), as in mi355x-vs-b200.md §5: conc -> (tok/s, per-user, TTFT s)
NOSPEC = {1: (86.7, 89.0, 0.23), 2: (165.2, 84.4, 0.218), 4: (280.8, 73.4, 0.233), 8: (475.3, 64.9, 0.23),
          16: (801.8, 53.6, 0.235), 32: (1209.3, 39.6, 0.264), 64: (1696.4, 27.9, 0.278)}
MI_C1 = 65.1                      # our MI355X vLLM recipe 1 per-user tok/s at c=1
f1 = lambda x: f"{x:,.1f}"


def load_arm(run, tag):
    d = os.path.join(run, tag)
    if not os.path.isdir(d):
        return None
    pts = {}
    for j in glob.glob(f"{d}/sweep/c*.json"):
        r = json.load(open(j))
        c = int(re.search(r"c(\d+)\.json$", j).group(1))
        if r.get("completed", 0) > 0 and r.get("median_tpot_ms"):
            pts[c] = dict(tps=r["output_throughput"], pu=1000 / r["median_tpot_ms"], tpot=r["median_tpot_ms"],
                          ttft=r["median_ttft_ms"], tot=r.get("total_token_throughput", 0), n=r["completed"])
    log = f"{d}/server/vllm_server.log"
    txt = open(log, errors="replace").read() if os.path.exists(log) else ""
    acc = [float(x) for x in re.findall(r"Mean acceptance length: ([\d.]+)", txt)]
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
    err = ""
    if os.path.exists(f"{d}/FAILED"):
        e = [l.strip() for l in txt.splitlines() if re.search(r"(Error|Exception):", l)]
        err = e[-1][-300:] if e else "see server log"
    return dict(pts=pts, acc=statistics.median(acc) if acc else None, kv=kv[-1] if kv else None, err=err)


def rep(s, old, new, what):
    if old not in s:
        print("WARN: text not found for", what)
        return s
    return s.replace(old, new, 1)


def add_col(s, header_start, nth, new_head, vals, pos=3):
    """insert a column at cell position pos (4 = after the 3rd cell, -1 = last) of the nth table
    whose header starts with header_start"""
    lines = s.split("\n")
    idx = [i for i, l in enumerate(lines) if l.startswith(header_start)]
    if len(idx) <= nth:
        print("WARN: table not found:", header_start, nth)
        return s
    i = idx[nth]
    def ins(line, cell):
        p = line.split("|")
        p.insert(pos if pos > 0 else len(p) - 1, f" {cell} ")
        return "|".join(p)
    lines[i] = ins(lines[i], new_head)
    lines[i + 1] = ins(lines[i + 1], "---:" if pos > 0 else "---")
    k = i + 2
    while k < len(lines) and lines[k].startswith("|"):
        c = lines[k].split("|")[1].strip()
        c = int(c) if c.isdigit() else c
        lines[k] = ins(lines[k], vals.get(c, "—"))
        k += 1
    return "\n".join(lines)


def turn(pt1, pt8, p, a, w):
    t1, u1, t8, u8 = pt1["ttft"] / 1000, pt1["pu"], pt8["ttft"] / 1000, pt8["pu"]
    P, A, W = t1 + (p - 1) / u1, t8 + (a - 1) / u8, t1 + (w - 1) / u1
    return P, A, W, P + A + W, P + A + t1, u1 * 0.75


def main():
    run, job = sys.argv[1].rstrip("/"), sys.argv[2]
    arms = {t: load_arm(run, t) for t, _, _ in ARMS}
    ok = {t: a for t, a in arms.items() if a and a["pts"]}
    kof = {"mtp7": 7, "mtp7d": 7, "mtp3": 3}
    s = open(MD).read()

    # --- B200-side summary ------------------------------------------------------------
    D = [f"# Kimi-K3 on 2 × 8 B200 with DSpark MTP (job {job})", "",
         "TP8 × PP2, `vllm/vllm-openai:kimi-k3`, draft `Inferact/Kimi-K3-DSpark` (with the `pard_token` shim), DCP 1, "
         "ISL/OSL 1024/1024, `--ignore-eos`, max-num-seqs 64, 10 × conc prompts. TPOT = time per output token, TTFT = time to first token. "
         "Per-user tok/s = 1000 / median TPOT. No-spec = job 20916742.", ""]
    for t, k, ab in ARMS:
        a = arms[t]
        if a is None:
            continue
        D.append(f"## {t}: draft length {k}" + (f", attention backend {ab}" if ab else ""))
        if a["err"]:
            D += ["", f"Server did not start: `{a['err']}`", ""]
            continue
        D += ["", f"Mean acceptance length (median of the server's log lines): {a['acc'] or '–'}; GPU KV cache: {a['kv'] or '–'} tokens.", "",
              "| Conc | out tok/s | per-user tok/s | TPOT med (ms) | TTFT med (ms) | no-spec per-user tok/s | MTP / no spec |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for c in sorted(a["pts"]):
            p = a["pts"][c]
            ns = NOSPEC.get(c)
            D.append(f"| {c} | {f1(p['tps'])} | {f1(p['pu'])} | {p['tpot']:.2f} | {p['ttft']:.0f} | "
                     f"{f1(ns[1]) if ns else '–'} | {p['pu'] / ns[1]:.2f}x |" if ns else
                     f"| {c} | {f1(p['tps'])} | {f1(p['pu'])} | {p['tpot']:.2f} | {p['ttft']:.0f} | – | – |")
        D.append("")
    open(f"{RES}/kimi-k3-mtp-b200.md", "w").write("\n".join(D) + "\n")

    if not ok:
        err = next((a["err"] for a in arms.values() if a and a["err"]), "no sweep point completed")
        s = rep(s, "**On B200 it was not attempted** — the speculator is not downloaded there (`HF_HUB_OFFLINE=1`).",
                f"**On B200 it was tried (job {job}, TP8 × PP2, DCP 1) but the server did not start:** `{err[:200]}`.", "B200 fail")
        open(MD, "w").write(s)
        line = f"B200 Kimi-K3 MTP (job {job}): no MTP server started; see {RES}/kimi-k3-mtp-b200.md"
        open(STATUS, "a").write(line + "\n"); print(line)
        return

    # best arm per conc, by per-user tok/s
    best = {}
    for t, a in ok.items():
        for c, p in a["pts"].items():
            if c not in best or p["pu"] > best[c][1]["pu"]:
                best[c] = (t, p)
    mark = lambda c: " (k3)" if kof[best[c][0]] == 3 else ""
    ks = sorted({kof[t] for t in ok}, reverse=True)
    klab = " and ".join(str(k) for k in ks)
    b1 = best.get(1, (None, None))[1]
    g1 = b1["pu"] / NOSPEC[1][1] if b1 else None

    s = rep(s, "- **16 × B200** (two nodes, TP8 × PP2), vLLM `vllm/vllm-openai:kimi-k3`, no speculative decoding.",
            f"- **16 × B200** (two nodes, TP8 × PP2), vLLM `vllm/vllm-openai:kimi-k3`: no speculative decoding (no spec), "
            f"and a second run with DSpark MTP (+MTP; `Inferact/Kimi-K3-DSpark`, draft length {klab}, job {job}).", "intro")
    s = rep(s, "| **Spec decoding (MTP)** | **B200** | **off** | both: no spec and **DSpark MTP on** | InferenceX's B200 +MTP arm gets a lever ours does not use — see §6 |",
            f"| **Spec decoding (MTP)** | **B200** | both: no spec and **DSpark MTP on** (draft length {klab}, DCP 1) | both: no spec and **DSpark MTP on** (draft length 7, DCP 8) | "
            f"same lever on both; our image does not allow DCP with the K3 DSpark draft — see §6 |", "§2 row")

    head = "| Conc | **Ours** B200 TP8×PP2 (no spec) |"
    s = add_col(s, head, 0, "**Ours** B200 TP8×PP2 +MTP", {c: f1(p["tps"]) + mark(c) for c, (t, p) in best.items()}, pos=4)  # right before InferenceX B200 +MTP
    s = add_col(s, head, 1, "**Ours** B200 TP8×PP2 +MTP", {c: f1(p["pu"]) + mark(c) for c, (t, p) in best.items()}, pos=4)  # right before InferenceX B200 +MTP
    note = (" (k3) = draft length 3; other +MTP points of ours on B200 use draft length 7." if 3 in ks and 7 in ks else "")
    s = rep(s, "**The InferenceX B200 no-spec column is not one curve.**",
            (f"Our B200 +MTP column: the faster of draft length {klab} at each point.{note}\n\n" if len(ks) > 1 else "")
            + "**The InferenceX B200 no-spec column is not one curve.**", "§3 note")

    if 64 in best:
        p = best[64][1]
        s = rep(s, "| **InferenceX** B200 +MTP, agentic | 32 |",
                f"| **Ours** B200 +MTP, 1024/1024 | 64 | {f1(p['tps'])} | {p['tps'] / 64:.1f} | {f1(p['pu'])} | **{p['tps'] / 64 / p['pu']:.2f}** |\n"
                "| **InferenceX** B200 +MTP, agentic | 32 |", "duty row")
    if b1:
        s = rep(s, "- **InferenceX's per-user lead is a real software lever** — MTP, which our B200 run does not use. Turning it on would raise *both* of our B200 columns (§6).",
                f"- **InferenceX's per-user lead is a real software lever** — MTP. Turning it on raises *both* of our B200 columns: "
                f"{g1:.1f}× per-user and {b1['tps'] / NOSPEC[1][0]:.1f}× aggregate at c=1 (§6).", "§3 bullet")
        L = s.split("\n")
        h = next((i for i, l in enumerate(L) if l.startswith("**What the table shows, in ")), None)
        if h is not None:
            j = h + 1
            while j < len(L) and not re.match(r"\d+\. ", L[j]):
                j += 1
            while j + 1 < len(L) and re.match(r"\d+\. ", L[j + 1]):
                j += 1
            n = int(re.match(r"(\d+)\.", L[j]).group(1)) + 1
            L.insert(j + 1, f"{n}. **MTP is worth {g1:.1f}× at c=1 on our B200** ({f1(b1['pu'])} vs 89.0), vs ~2.7× in the InferenceX data. "
                            f"With MTP on both, our B200 streams {b1['pu'] / MI_C1:.2f}× as fast as our MI355X vLLM recipe 1 at c=1 ({f1(b1['pu'])} vs {MI_C1}).")
            L[h] = re.sub(r"in \w+ points", "in " + {4: "four", 5: "five", 6: "six"}.get(n, str(n)) + " points", L[h])
            s = "\n".join(L)

    if 1 in best and 8 in best:
        rows = []
        for (p, a, w), anchor in (((200, 500, 800), "| **Ours** MI355X vLLM recipe 1 (+MTP) | 3.3 s |"),
                                  ((30, 70, 100), "| **Ours** MI355X vLLM recipe 1 (+MTP) | 0.6 s |")):
            P, A, W, T, F, R = turn(best[1][1], best[8][1], p, a, w)
            s = rep(s, anchor, f"| **Ours** B200 +MTP | {P:.1f} s | {A:.1f} s | {W:.1f} s | **{T:.1f} s** | {F:.1f} s | {R:.0f} words/s |\n" + anchor, f"turn {p}")
            rows.append(R)
        m = re.search(r"Every row prints at (\d+)–(\d+) words/s", s)
        if m:
            lo, hi = min(int(m.group(1)), *rows), max(int(m.group(2)), *rows)
            s = s.replace(m.group(0), f"Every row prints at {lo:.0f}–{hi:.0f} words/s", 1)

    g = [(c, best[c][1]["pu"], NOSPEC[c][1]) for c in (1, 8, 32) if c in best]
    if g:
        tab = ["", f"Measured gain in our B200 data (MTP vs no-spec, same hardware, 1024/1024, job {job}):", "",
               "| | " + " | ".join(f"c={c}" for c, _, _ in g) + " |", "|---|" + "---:|" * len(g),
               "| B200 per-user tok/s, **no spec** | " + " | ".join(f1(n) for _, _, n in g) + " |",
               "| B200 per-user tok/s, **MTP** | " + " | ".join(f1(m) for _, m, _ in g) + " |",
               "| ratio | " + " | ".join(f"**{m / n:.1f}×**" for _, m, n in g) + " |", ""]
        s = rep(s, "\nCeiling is draft length × acceptance rate", "\n".join(tab) + "\nCeiling is draft length × acceptance rate", "§6 gain table")
    accs = "; ".join(f"{a['acc']:.2f} at draft length {kof[t]}" for t, a in ok.items() if a["acc"])
    if accs:
        line = next((l for l in s.split("\n") if l.startswith("Ceiling is draft length")), None)
        if line:
            s = s.replace(line, line + f" In our B200 +MTP run it is {accs} (`../../b200-kimi/results/kimi-k3-mtp-b200.md`).", 1)

    ab = "`TOKENSPEED_MLA`" if "mtp7" in ok else "default"
    s = add_col(s, "| Step | MI355X (our vLLM recipe 1) |", 0, "B200 (ours)", {}, pos=-1)
    for old, new in (("| 1. Speculator |", "`Inferact/Kimi-K3-DSpark` (same file as MI355X)"),
                     ("| 2. `pard_token` shim |", "**required** (added)"),
                     ("| 3. Extra server flags |", "none (DCP is not allowed with this draft in our image)"),
                     ("| 4. Attention backend in the config |", ab),
                     ("| 5. Draft length |", klab)):
        line = next((l for l in s.split("\n") if l.startswith(old)), None)
        if line:
            p = line.split("|"); p[-2] = f" {new} "
            s = s.replace(line, "|".join(p), 1)
    s = rep(s, "**On B200 it was not attempted** — the speculator is not downloaded there (`HF_HUB_OFFLINE=1`). "
               "Given the ~2.7× per-user gain at c=1 measured above, this is the single highest-value follow-up available on B200.",
            f"**On B200 it is on in our +MTP run** (job {job}): TP8 × PP2 with the `pard_token` shim, but DCP 1 — this image refuses "
            f"DCP with the K3 DSpark draft (*\"MLA DSpark does not currently support decode context parallelism\"*), so the three "
            f"extra flags above are not used. " + (f"It gives {g1:.1f}× per-user tok/s at c=1 (§4)." if g1 else ""), "§6 B200")
    open(MD, "w").write(s)
    line = (f"B200 Kimi-K3 MTP (job {job}): arms {', '.join(ok)}; c=1 per-user {f1(b1['pu'])} tok/s ({g1:.1f}x no spec)"
            if b1 else f"B200 Kimi-K3 MTP (job {job}): arms {', '.join(ok)}") + f"; {MD} updated"
    open(STATUS, "a").write(line + "\n"); print(line)


if __name__ == "__main__":
    main()
