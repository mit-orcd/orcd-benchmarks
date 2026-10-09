#!/usr/bin/env python3
"""Spell out TPOT (time per output token) and TTFT (time to first token) where each first
appears in each results md file. Prose line: "TPOT" -> "TPOT (time per output token)". Table
line: a line "TPOT = time per output token." is put just above the table. Files that already
spell the term out before its first use are left alone. Usage: expand_terms.py [files or dirs (md, depth<=4)]"""
import re, sys
from pathlib import Path

TERMS = {"TPOT": "time per output token", "TTFT": "time to first token"}

def fix(p, T, FULL):
    L = p.read_text().split("\n")
    for i, l in enumerate(L):
        if re.search(rf"\b{T}\b", l):
            break
    else:
        return False
    if FULL in "\n".join(L[: i + 1]).lower():
        return False
    k0 = re.search(rf"\b{T}\b", l).start()
    if l.lstrip().startswith("|") or l[:k0].count("`") % 2:
        j = i
        while j > 0 and L[j - 1].lstrip().startswith("|"):
            j -= 1
        L[j:j] = [f"{T} = {FULL}.", ""]
    else:
        k = re.search(rf"\b{T}\b", l).start()
        inside = l[:k].count("(") > l[:k].count(")")
        if not inside:
            rep = f"{T} ({FULL})"
        elif re.match(r"\s*[\d(]", l[k + len(T):]):
            rep = f"{T} = {FULL}:"
        else:
            rep = f"{T} = {FULL}"
        L[i] = l[:k] + rep + l[k + len(T):]
    p.write_text("\n".join(L))
    return True

def files(args):
    for a in args:
        a = Path(a)
        if a.is_dir():
            for d in range(1, 5):
                yield from sorted(a.glob("/".join(["*"] * d) + ".md"))
        elif a.suffix == ".md":
            yield a

if __name__ == "__main__":
    n = sum(fix(p, T, F) for p in files(sys.argv[1:] or ["/orcd/data/orcd/022/benchmarks/amd-ubuntu/results"]) for T, F in TERMS.items())
    print(f"expand_terms: {n} files updated")
