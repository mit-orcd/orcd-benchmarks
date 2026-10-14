#!/usr/bin/env python3
"""Analysis of atom/run_kimi_prefix_test.sh (vLLM recipe 2, 128K/1K, cold vs repeated prompts).

  python3 analyze_prefix_test.py [<run dir>]

Merges the tests.txt of every prefix-test run (logs/node*/kimi-amd-recipe/atom/kimi_prefix_test_*;
a later run replaces an earlier one's test of the same tag), so the 1-4 user run and the 8-128
user runs on the two nodes give one table. Also fills the prefix-cache column of table 7.3.

Writes results/ubuntu/kimi-prefix-cache-test.md and replaces the marker blocks in
results/kimi-summary.md (before "## 3. ") and results/mi355x-vs-b200.md (before "### 7.4 ").
tests.txt lines: tag|concurrency|prompts|seed|prefix-cache queries|hits|result json.
"""
import datetime, glob, json, os, re, sys

BENCH = "/orcd/data/orcd/022/benchmarks/amd-ubuntu"
RES = os.environ.get("PREFIX_RES", f"{BENCH}/results")   # override for tests
GPUS = 8
AMD = {1: 874, 2: 1090, 4: 1235, 8: 1199, 16: 1179, 32: 1118, 64: 1096, 128: 1110}  # amd-kimi-k3-recipe.pdf, total tok/s/GPU
OURS_OLD = {1: 487, 2: 791, 4: 1015, 8: 1212, 16: 1189, 32: 1227, 64: 1191, 128: 1188}  # our 2026-10-07 sweep (0% hits)
KIND = {"cold": 0, "repeat": 1, "seed1": 2}
BEGIN = "<!-- PREFIX-TEST-BEGIN (auto: atom/analyze_prefix_test.py) -->"
END = "<!-- PREFIX-TEST-END -->"


def runs():
    """All prefix-test run dirs with a tests.txt, oldest first (by the timestamp in the name)."""
    ds = glob.glob(f"{BENCH}/logs/node*/kimi-amd-recipe/atom/kimi_prefix_test_*/tests.txt")
    return sorted((os.path.dirname(d) for d in ds), key=lambda d: d.rsplit("_", 2)[-2:])


def load_all(extra=None):
    t, src = {}, []
    ds = runs()
    if extra and extra not in ds:
        ds.append(extra)
    for d in ds:
        u = load(d)
        if u:
            t.update(u)
            src.append(d)
    order = sorted(t, key=lambda k: (t[k]["c"], KIND.get(k.split("_", 1)[1], 9)))
    return {k: t[k] for k in order}, src


def load(run):
    t = {}
    if not os.path.exists(os.path.join(run, "tests.txt")):
        return t
    for line in open(os.path.join(run, "tests.txt")):
        if line.count("|") != 6:
            continue
        tag, c, n, seed, q, h, f = line.strip().split("|")
        p = os.path.join(run, f)
        d = json.load(open(p)) if os.path.exists(p) else {}
        q, h = int(q), int(h)
        t[tag] = dict(c=int(c), n=int(n), seed=int(seed), hit=(h / q if q > 0 else None),
                      tps=d.get("total_token_throughput", 0) / GPUS if d else None,
                      ttft=d.get("median_ttft_ms"), tpot=d.get("median_tpot_ms"),
                      ok=d.get("completed", 0), fail=d.get("failed", 0))
    return t


def f0(x, nd=0, suf=""):
    return "–" if x is None else f"{x:,.{nd}f}{suf}"


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def verdict(t):
    """One line per user count + overall verdict for 1 user."""
    lines, overall = [], "not tested"
    for c in sorted({r["c"] for r in t.values()}):
        cold, rep = t.get(f"c{c}_cold"), t.get(f"c{c}_repeat")
        if not cold or not rep or cold["tps"] is None or rep["tps"] is None:
            lines.append(f"- **{c} user{'s' if c > 1 else ''}:** no result.")
            continue
        r_amd, r_cold = rep["tps"] / AMD[c], rep["tps"] / cold["tps"]
        hit = rep["hit"]
        if hit is not None and hit >= 0.5 and r_amd >= 1.1:
            v = "the repeated run is faster than AMD's number"
        elif hit is not None and hit >= 0.5 and r_amd >= 0.85:
            v = "the repeated run reaches AMD's number"
        elif hit is not None and hit >= 0.5 and r_cold >= 1.2:
            v = "the repeated run is faster, but still below AMD's number"
        elif hit is not None and hit < 0.1:
            v = "the repeated prompts were not found in the cache, so the cache cannot explain the gap here"
        else:
            v = "the repeated run is not clearly faster"
        lines.append(f"- **{c} user{'s' if c > 1 else ''}:** repeated run {f0(rep['tps'])} vs first run "
                     f"{f0(cold['tps'])} tok/s/GPU ({r_cold:.2f}x), {r_amd:.2f}x AMD's {AMD[c]:,}; "
                     f"prefix-cache hit rate {pct(hit)}: {v}.")
        if c == 1:
            if hit is not None and hit >= 0.5 and r_amd >= 0.85:
                overall = "confirmed"
            elif hit is not None and hit >= 0.5 and r_cold >= 1.2:
                overall = "partly confirmed"
            else:
                overall = "not confirmed"
    return overall, lines


def new_prompts(t):
    s = (t.get("c1_seed1") or {}).get("tps")
    if s is None:
        return ""
    if abs(s / OURS_OLD[1] - 1) <= 0.1:
        return f"With new prompts (another seed) it drops back to {f0(s)}, as in our earlier run ({OURS_OLD[1]}). "
    return f"With new prompts (another seed) it gives {f0(s)} (earlier run: {OURS_OLD[1]}). "


def high_users(t):
    """One sentence on the 8+ user tests, if they ran."""
    hi = [c for c in sorted({r["c"] for r in t.values()}) if c >= 8
          and (t.get(f"c{c}_repeat") or {}).get("tps") and (t.get(f"c{c}_cold") or {}).get("tps")]
    if not hi:
        return ""
    hits = [t[f"c{c}_repeat"]["hit"] or 0 for c in hi]
    rc = [t[f"c{c}_repeat"]["tps"] / t[f"c{c}_cold"]["tps"] for c in hi]
    ra = [t[f"c{c}_repeat"]["tps"] / AMD[c] for c in hi]
    users = f"{hi[0]}–{hi[-1]}" if len(hi) > 1 else f"{hi[0]}"
    def rg(x, f):
        a, b = f.format(min(x)), f.format(max(x))
        return a if a == b else f"{a}–{b}"
    return (f" At {users} users the repeated run gets {rg([100 * h for h in hits], '{:.0f}')}% cache hits and "
            f"{rg(rc, '{:.2f}')}x the first run, {rg(ra, '{:.2f}')}x AMD's numbers"
            + (": the prompts do not fit in the KV cache, so the cache plays no role there."
               if max(hits) < 0.1 else "."))


def headline(overall, t):
    rep = t.get("c1_repeat") or {}
    if overall == "confirmed":
        return ("**Confirmed: prompts reused from the prefix cache explain the 1-user gap.** Running the same "
                f"prompts again on the same server gives {f0(rep.get('tps'))} tok/s/GPU at 1 user, "
                f"{rep['tps'] / AMD[1]:.2f}x AMD's 874, with first-token time down to {f0(rep.get('ttft'))} ms. "
                + new_prompts(t) +
                "So AMD's 1-user number most likely includes cache hits, and our nodes are almost the same "
                "as AMD's for vLLM recipe 2.")
    if overall == "partly confirmed":
        s = ("**Partly confirmed: the prefix cache explains part of the low-user gap.** Repeated prompts "
             f"give {f0(rep.get('tps'))} tok/s/GPU at 1 user ({rep['tps'] / AMD[1]:.2f}x AMD's 874), "
             "faster than with new prompts but still below AMD's number")
        r2, c4 = t.get("c2_repeat") or {}, t.get("c4_repeat") or {}
        if r2.get("tps"):
            s += f", and {f0(r2['tps'])} at 2 users ({r2['tps'] / AMD[2]:.2f}x AMD's 1,090)"
        s += ". AMD's 1- and 2-user numbers lie between our first and repeated runs, which fits runs with partly cached prompts."
        if c4.get("tps") and (c4.get("hit") or 0) < 0.1:
            s += (f" At 4 users the 40 prompts do not fit in the KV cache, so nothing is reused (also not in AMD's runs), "
                  f"and we stay at {c4['tps'] / AMD[4]:.2f}x AMD's number.")
        s += high_users(t)
        if rep.get("tpot"):
            s += f" At 1 user the TPOT is {f0(rep['tpot'], 1)} ms with or without the cache."
        return s + " So part of the gap has another cause (still to be profiled)."
    if overall == "not confirmed":
        return ("**Not confirmed: the prefix cache does not explain the 1-user gap.** Repeating the same "
                "prompts on the same server does not bring our 1-user number close to AMD's 874 tok/s/GPU, "
                "so the gap has another cause (still to be profiled).")
    return "**The test did not produce a 1-user result** (see the run log)."


def table(t):
    rows = ["| test | users | prompts | seed | prompts seen before? | prefix-cache hit rate | "
            "TTFT median, ms | TPOT median, ms | total tok/s/GPU | vs AMD tested |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
    seen = {"cold": "no (first run)", "repeat": "**yes (same prompts again)**", "seed1": "no (new seed)"}
    for tag, r in t.items():
        kind = tag.split("_", 1)[1]
        amd = AMD.get(r["c"])
        ra = f"{r['tps'] / amd:.2f}x" if r["tps"] and amd else "–"
        rows.append(f"| `{tag}` | {r['c']} | {r['n']} | {r['seed']} | {seen.get(kind, kind)} | {pct(r['hit'])} | "
                    f"{f0(r['ttft'])} | {f0(r['tpot'], 1)} | {f0(r['tps'])} | {ra} |")
    return rows


def fill_table_73(t, path):
    """Prefix-cache column + ratio column of table 7.3 in mi355x-vs-b200.md (only users with a result)."""
    s = open(path).read()
    a = s.find("### 7.3 ")
    b = s.find(BEGIN, a) if a >= 0 else -1
    if a < 0 or b < 0:
        print("WARN: table 7.3 not found; not filled"); return
    sec, out = s[a:b], []
    head = None
    for l in sec.split("\n"):
        c = [x.strip() for x in l.split("|")]
        if l.startswith("| Users |"):
            head = c
        elif head and l.startswith("| ") and len(c) == len(head) and c[1].isdigit():
            u = int(c[1]); r = (t.get(f"c{u}_repeat") or {}).get("tps")
            ip = next(i for i, h in enumerate(head) if h.startswith("ours (vLLM recipe 2, prefix-cache"))
            ir = next(i for i, h in enumerate(head) if h.startswith("(prefix-cache test)"))
            if r:
                c[ip], c[ir] = f"{r:,.0f}", f"{r / AMD[u]:.2f}x"
            l = "| " + " | ".join(c[1:-1]) + " |"
        out.append(l)
    sec = "\n".join(out)
    hit = [(r["c"], r["hit"]) for k, r in t.items() if k.endswith("_repeat") and r["hit"] is not None]
    on = [c for c, h in hit if h >= 0.5]; off = [c for c, h in hit if h < 0.1]
    def us(x):
        return (", ".join(map(str, x[:-1])) + " and " if len(x) > 1 else "") + str(x[-1]) + (" user" if x == [1] else " users")
    txt = ("- **Prefix-cache test column:** vLLM recipe 2 run a second time with the same prompts on the same server "
           f"(prefix-cache hit rate about 100% at {us(on)}" if on else "- **Prefix-cache test column:** vLLM recipe 2 run a second time with the same prompts on the same server (")
    if off:
        txt += f"{', ' if on else ''}0% at {us(off)}, where the prompts do not fit in the KV cache"
    txt += "); 2026-10-13/14, see the test below."
    sec = re.sub(r"^- \*\*Prefix-cache test column:\*\*.*$", lambda _: txt, sec, flags=re.M)
    open(path, "w").write(s[:a] + sec + s[b:])
    print("filled table 7.3 in", path)


def put_block(path, block, before):
    s = open(path).read()
    s = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n*", "", s, flags=re.S)
    i = s.find(before)
    if i < 0:
        print(f"WARN: '{before.strip()}' not found in {path}; block not inserted")
        return
    s = s[:i] + BEGIN + "\n" + block.strip() + "\n" + END + "\n\n" + s[i:]
    open(path, "w").write(s)
    print("updated", path)


def main():
    t, src = load_all(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else None)
    if not t:
        sys.exit("no prefix-test results found")
    overall, vlines = verdict(t)
    head = headline(overall, t)
    day = datetime.date.today()
    rel = "`, `".join(os.path.relpath(d, BENCH) for d in src)

    md = ["# Prefix-cache test: vLLM recipe 2 on AMD's 128K/1K workload", "",
          f"Generated {day} by `atom/analyze_prefix_test.py` from `{rel}` (script `atom/run_kimi_prefix_test.sh`).",
          "",
          "**Question.** With vLLM recipe 2 (`amd-kimi-k3-recipe.pdf`) our nodes match AMD's tested numbers from 8 users up "
          "but reach only 0.56x, 0.73x and 0.82x of them at 1, 2 and 4 users. The PDF's client sets no `--seed` (so every run "
          "sends the same prompts) and vLLM v0.29.0 keeps prefix caching on by default; 10–20 prompts of about 122K tokens fit "
          "in the 2.8M-token KV cache. If AMD's low-user runs followed an earlier run with the same prompts, most of the "
          "prompt processing would have come from the cache.",
          "",
          "**Test.** Same image, server flags and client flags as our earlier run, one server: each user count is run "
          "twice with the same prompts (first run, then repeated run), and 1 user once more with new prompts. The prefix-cache "
          "hit rate is read from the server's counters before and after each test. 8–128 users were added "
          "later (2026-10-14, one server per node, new prompts per user count), to fill the table. TTFT = time to first token, "
          "TPOT = time per output token; metric = total tokens (input + output) per second per GPU, as in the PDF.",
          "",
          "## Result", "",
          f"**Compared: the repeated run (same prompts again) vs the first run and vs AMD's tested numbers, same node, recipe and workload.** {head}",
          ""] + vlines + ["",
          "| users | AMD tested | ours, 2026-10-07 sweep | ours, first run | ours, repeated run |",
          "|---:|---:|---:|---:|---:|"]
    for c in AMD:
        cold, rep = t.get(f"c{c}_cold", {}), t.get(f"c{c}_repeat", {})
        md.append(f"| {c} | {AMD[c]:,} | {OURS_OLD[c]:,} | {f0(cold.get('tps'))} | {f0(rep.get('tps'))} |")
    md += ["", "## All tests", "",
           "**Compared: each test vs AMD's tested number at the same users; ratio above 1 = ours faster.**", ""] + table(t) + ["",
           "- 4 users use 40 prompts (about 4.9M tokens), more than the KV cache holds, so few hits are expected there "
           "even when prompts repeat.",
           "- A high hit rate with a short TTFT means the prompt was mostly read from the cache instead of being processed.",
           "",
           "## What it means", "",
           "- Prompts reused from the cache can make a 1–2 user number look much faster (1.5–1.9x here), so this part of "
           "the gap to `amd-kimi-k3-recipe.pdf` is a benchmarking effect, not a slower node. Fair comparisons should use "
           "new prompts for every run (a different `--seed`) or restart the server between runs.",
           "- The cache does not explain everything: at 1 user (fully cached) and at 4 users (no reuse possible) we are "
           "still at about 0.82x AMD's number. This remaining gap is in the decode steps (TPOT) and is still to be "
           "profiled (e.g. rocprofv3).",
           "- From 8 users up the prompts (80 or more of about 128K tokens) do not fit in the 2.8M-token KV cache, "
           "so a repeated run cannot reuse them and the cache does not change those numbers.",
           "- It is still worth asking AMD whether their table was measured after an earlier run on the same server.", ""]
    out = f"{RES}/ubuntu/kimi-prefix-cache-test.md"
    open(out, "w").write("\n".join(md))
    print("wrote", out)

    short = (f"**Prefix-cache test ({day}):** {head} Detail: "
             "[ubuntu/kimi-prefix-cache-test.md](ubuntu/kimi-prefix-cache-test.md).\n\n" + "\n".join(vlines))
    put_block(f"{RES}/kimi-summary.md", short, "## 3. vLLM recipe 2 on our workload")
    put_block(f"{RES}/mi355x-vs-b200.md", short, "### 7.4 Apple-to-apple")
    fill_table_73(t, f"{RES}/mi355x-vs-b200.md")


if __name__ == "__main__":
    main()
