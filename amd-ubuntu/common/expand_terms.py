#!/usr/bin/env python3
"""Spell out TPOT (time per output token) where it first appears in each results md file.
Prose line: "TPOT" -> "TPOT (time per output token)". Table line: a line "TPOT = time per
output token." is put just above the table. Files that already say "time per output token"
before the first TPOT are left alone. Usage: expand_terms.py [files or dirs (md, depth<=4)]"""
import re, sys
from pathlib import Path

FULL = "time per output token"

def fix(p):
    L = p.read_text().split("\n")
    for i, l in enumerate(L):
        if re.search(r"\bTPOT\b", l):
            break
    else:
        return False
    if FULL in "\n".join(L[: i + 1]).lower():
        return False
    k0 = re.search(r"\bTPOT\b", l).start()
    if l.lstrip().startswith("|") or l[:k0].count("`") % 2:
        j = i
        while j > 0 and L[j - 1].lstrip().startswith("|"):
            j -= 1
        L[j:j] = ["TPOT = time per output token.", ""]
    else:
        k = re.search(r"\bTPOT\b", l).start()
        inside = l[:k].count("(") > l[:k].count(")")
        if not inside:
            rep = "TPOT (time per output token)"
        elif re.match(r"\s*[\d(]", l[k + 4:]):
            rep = "TPOT = time per output token:"
        else:
            rep = "TPOT = time per output token"
        L[i] = l[:k] + rep + l[k + 4:]
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
    n = sum(fix(p) for p in files(sys.argv[1:] or ["/orcd/data/orcd/022/benchmarks/amd-ubuntu/results"]))
    print(f"expand_terms: {n} files updated")
