#!/usr/bin/env python3
"""Nov 12 rerun: check one Hyperloom session for the problems found in session A.

  check_session.py <session dir> [run log]

Prints markdown lines, then a last line "RESULT PASS" or "RESULT FAIL: <reasons>".
Checks: (1) JSON quoting (no "invalid loads value"), (2) temp folder (no Errno 39),
(3) explore variants that produced a benchmark number, (4) helpers could read code
(no sandbox/permission refusals, claude CLI found, no in-process fallback),
(5) conc sweep (multi-user check) outcome. Read-only.
"""
import glob
import json
import os
import re
import sys

S = sys.argv[1]
LOG = sys.argv[2] if len(sys.argv) > 2 else ""
R = os.path.join(S, "runs")


def grep_files(pattern, rx):
    hits = []
    for f in glob.glob(os.path.join(R, "**", pattern), recursive=True):
        try:
            with open(f, errors="replace") as fh:
                if re.search(rx, fh.read()):
                    hits.append(f)
        except OSError:
            pass
    return hits


def tput(d):
    best = 0.0
    for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
        try:
            t = open(f, errors="replace").read()
        except OSError:
            continue
        for m in re.finditer(r'"output_throughput"\s*:\s*([0-9.]+)', t):
            best = max(best, float(m.group(1)))
    return best


out, fails = [], []
loads = grep_files("server.log", r"invalid loads value")
errno39 = grep_files("*.log", r"Errno 39") + grep_files("abort_reason.json", r"Errno 39")
out.append(f"- JSON quoting: {len(loads)} server logs with `invalid loads value`")
out.append(f"- Temp folder: {len(errno39)} files with `Errno 39`")
if loads:
    fails.append("JSON quoting still broken")
if errno39:
    fails.append("temp-folder clash still there")

ex = sorted(glob.glob(os.path.join(R, "explore", "*", "v*")))
ok = [(os.path.basename(v), tput(v)) for v in ex]
good = [v for v in ok if v[1] > 0]
out.append(f"- Explore: {len(good)} of {len(ex)} variants gave a benchmark number"
           + (": " + ", ".join(f"{n} {t:,.0f} tok/s" for n, t in good[:8]) if good else ""))
if not ex:
    fails.append("explore did not run")
elif not good:
    fails.append("no explore variant gave a number")

sp = sorted(glob.glob(os.path.join(R, "specialist", "*", "specialist_done.json")))
blocked = []
for f in sp:
    try:
        t = json.dumps(json.load(open(f)))
    except Exception:
        continue
    if re.search(r"sandbox|permission denied|denied all reads|could not read|cannot read|reads? (were |was )?(blocked|refused)|(blocked|refused) (all )?reads?", t, re.I):
        blocked.append(os.path.basename(os.path.dirname(f)))
logtxt = open(LOG, errors="replace").read() if LOG and os.path.exists(LOG) else ""
fallback = "falling back to in-process" in logtxt
missing = "claude CLI for helpers: MISSING" in logtxt
cli = re.search(r"claude CLI for helpers: (\S+)", logtxt)
out.append(f"- Helpers: {len(sp)} finished, {len(blocked)} report blocked reads"
           f"; claude CLI {cli.group(1) if cli else '?'}; in-process fallback {'yes' if fallback else 'no'}")
if not sp:
    fails.append("no helper finished")
if blocked:
    fails.append(f"helpers still blocked ({', '.join(blocked[:3])})")
if fallback or missing:
    fails.append("helpers did not use the claude CLI")

cs = sorted(glob.glob(os.path.join(R, "conc_sweep", "*", "variant_*")))
ab = []
for v in cs:
    f = os.path.join(v, "abort_reason.json")
    if os.path.exists(f):
        try:
            ab.append(json.load(open(f)).get("reason") or json.load(open(f)).get("abort_reason") or "aborted")
        except Exception:
            ab.append("aborted")
csok = [os.path.basename(v) for v in cs if tput(v) > 0]
out.append(f"- Multi-user check (conc sweep): {len(csok)} of {len(cs)} levels gave a number"
           + (f"; aborts: {', '.join(sorted(set(map(str, ab))))}" if ab else ""))

fin = os.path.join(S, "reports", "final.md")
if os.path.exists(fin):
    lines = [l.strip() for l in open(fin, errors="replace") if re.search(r"baseline|best|gain", l, re.I)]
    for l in lines[:4]:
        out.append(f"  - final.md: {l[:200]}")
try:
    st = json.load(open(os.path.join(S, "state.json")))
    out.append(f"- stop_reason: {st.get('stop_reason')}")
except Exception:
    pass

print("\n".join(out))
print("RESULT PASS" if not fails else "RESULT FAIL: " + "; ".join(fails))
