#!/usr/bin/env python3
"""Greedy (temperature 0) output for fixed prompts, to compare MTP (k>0) with no MTP (k=0).
   greedy_capture.py <base url> <out.json>"""
import json, sys, urllib.request
PROMPTS = [
    "Explain how a CPU cache works, step by step.",
    "Write a Python function that merges two sorted lists, then explain its running time.",
    "List the planets of the solar system and give one fact about each.",
    "What causes the seasons on Earth? Answer in three paragraphs.",
    "Prove that the square root of 2 is irrational.",
    "Summarize the main differences between TCP and UDP.",
    "Translate into French: The library opens at nine and closes at five on weekdays.",
    "Describe how a bill becomes a law in the United States.",
]
base, out = sys.argv[1], sys.argv[2]
res = []
for p in PROMPTS:
    body = json.dumps({"model": "/model", "prompt": p, "max_tokens": 256,
                       "temperature": 0, "top_p": 1, "seed": 1234}).encode()
    r = urllib.request.Request(base + "/v1/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1800) as f:
        d = json.load(f)
    res.append({"prompt": p, "text": d["choices"][0]["text"], "usage": d.get("usage")})
json.dump(res, open(out, "w"), indent=1)
print(f"greedy: {len(res)} prompts -> {out}")
