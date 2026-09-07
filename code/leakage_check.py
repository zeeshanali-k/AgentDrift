#!/usr/bin/env python3
"""
World-identity leakage check (README, 'World-identity leakage').

Assigns each test trajectory the majority category of the identical world object in the
training part of the same split (benign if unseen), with no model at all. High accuracy
means a text model can learn category from identity rather than behavior.

Usage:
    python code/leakage_check.py --root data --split stratified
    python code/leakage_check.py --root data --split task_disjoint
"""
import json
import argparse
from collections import defaultdict, Counter

from data_loader import load_trajectories

DOMAINS = ["email", "banking", "web", "coding", "medical"]


def key(t):
    return json.dumps(t["world"], sort_keys=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="stratified", choices=["stratified", "task_disjoint"])
    args = ap.parse_args()
    tr = load_trajectories(args.root, "train", split=args.split)
    te = load_trajectories(args.root, "test", split=args.split)
    lookup = defaultdict(Counter)
    for t in tr:
        lookup[key(t)][t["category"]] += 1
    res = defaultdict(lambda: [0, 0, 0, 0])  # seen, 4-class correct, binary correct, total
    for t in te:
        k = key(t)
        pred = lookup[k].most_common(1)[0][0] if k in lookup else "benign"
        r = res[t["agent"]]
        r[3] += 1
        r[0] += k in lookup
        r[1] += pred == t["category"]
        r[2] += (pred == "attacked") == (t["category"] == "attacked")
    tot = [0, 0, 0, 0]
    print(f"split={args.split}  majority-class binary rate={1 - sum(t['category']=='attacked' for t in te)/len(te):.3f}")
    print(f"{'domain':10s} {'test':>5s} {'seen':>5s} {'4-class':>8s} {'binary':>8s}")
    for d in DOMAINS:
        r = res[d]
        tot = [x + y for x, y in zip(tot, r)]
        print(f"{d:10s} {r[3]:5d} {r[0]:5d} {r[1]/r[3]:8.3f} {r[2]/r[3]:8.3f}")
    print(f"{'all':10s} {tot[3]:5d} {tot[0]:5d} {tot[1]/tot[3]:8.3f} {tot[2]/tot[3]:8.3f}")


if __name__ == "__main__":
    main()
