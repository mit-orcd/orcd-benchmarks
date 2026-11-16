#!/usr/bin/env python3
"""Write into hyperloom/rerun-report.md (Nov 12 rerun) under a file lock.

  rerun_report.py add <A|B> "<markdown line>"     append a line to part A or B
  rerun_report.py set <A|B> <file.md>             replace part A or B with the file's text
Parts are kept between <!-- X-BEGIN --> and <!-- X-END --> markers, so the node6100 pipeline (A)
and the node6101 candidates (B) can write at the same time.
"""
import fcntl
import os
import sys
import datetime

REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rerun-report.md")
LOCK = "/orcd/data/orcd/022/benchmarks/amd-ubuntu/logs/hyperloom/rerun-report.lock"


def main() -> int:
    op, part = sys.argv[1], sys.argv[2]
    b, e = f"<!-- {part}-BEGIN -->", f"<!-- {part}-END -->"
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        text = open(REPORT).read()
        i, j = text.index(b) + len(b), text.index(e)
        if op == "add":
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%b %d %H:%M UTC")
            body = text[i:j].rstrip("\n") + f"\n- {ts}: {sys.argv[3]}\n"
        else:
            body = "\n" + open(sys.argv[3]).read().strip("\n") + "\n"
        tmp = REPORT + ".tmp"
        with open(tmp, "w") as f:
            f.write(text[:i] + body + text[j:])
        os.replace(tmp, REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
