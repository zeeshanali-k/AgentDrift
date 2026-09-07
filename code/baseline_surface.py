#!/usr/bin/env python3
"""
AgentDrift surface-feature baseline (benchmark validation).

Six trajectory-level surface features (no sequence modeling, no embeddings) fed to a
logistic regression with balanced class weights. Reproduces the numbers in the dataset
paper: on the stratified test split, attacked-class P 0.778 / R 0.554 / F1 0.647.

Usage:
    python code/baseline_surface.py --root data --split stratified
    python code/baseline_surface.py --root data --split task_disjoint
"""

import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data_loader import (load_trajectories, world_known_sets,
                         POSITIVE_CATEGORIES, EMAIL_RE, URL_RE)

KEYWORDS = [
    "security notice", "per policy", "ignore previous", "ignore all",
    "you must", "must forward", "must send", "urgent", "immediately",
    "external", "audit", "verify your", "confirm your", "click here",
    "login", "password", "transfer", "new address", "updated address",
    "compliance", "regulation", "as instructed", "system update",
    "forward all", "send all", "delete all",
]
SINK_TOOLS = ["delete", "send", "forward", "execute", "transfer", "post",
              "share", "write_file", "email"]


def featurize(traj):
    """Crude surface feature vector for one trajectory. No sequence info."""
    steps = traj["steps"]
    known_emails, known_domains = world_known_sets(traj.get("world", {}))

    ext_email_total, ext_url_any, sink_count = 0, 0, 0
    text_blob = []
    for s in steps:
        try:
            args_blob = json.dumps(s.get("args", {}), ensure_ascii=False).lower()
        except Exception:
            args_blob = str(s.get("args", {})).lower()
        emails = EMAIL_RE.findall(args_blob)
        ext = [e for e in emails
               if e.split("@")[-1] not in known_domains and e not in known_emails]
        ext_email_total += len(ext)
        urls = URL_RE.findall(args_blob)
        if any(not any(d and d in u for d in known_domains) for u in urls):
            ext_url_any = 1
        tool = (s.get("tool", "") or "").lower()
        if any(k in tool for k in SINK_TOOLS):
            sink_count += 1
        text_blob.append((s.get("obs", "") or "").lower())
        text_blob.append((s.get("thought", "") or "").lower())

    blob = " ".join(text_blob)
    kw_hits = sum(1 for k in KEYWORDS if k in blob)

    return [
        len(steps),
        ext_email_total,
        1.0 if ext_email_total > 0 else 0.0,
        ext_url_any,
        sink_count,
        kw_hits,
    ]


def build_xy(trajs):
    X = np.array([featurize(t) for t in trajs], dtype=float)
    y = np.array([1 if t.get("category") in POSITIVE_CATEGORIES else 0
                  for t in trajs], dtype=int)
    cats = [t.get("category") for t in trajs]
    return X, y, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="stratified", choices=["stratified", "task_disjoint"])
    args = ap.parse_args()

    tr = load_trajectories(args.root, "train", split=args.split)
    te = load_trajectories(args.root, "test", split=args.split)
    Xtr, ytr, _ = build_xy(tr)
    Xte, yte, cats_te = build_xy(te)

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(scaler.transform(Xtr), ytr)
    pred = clf.predict(scaler.transform(Xte))

    # overall (attacked = positive)
    tp = int(((pred == 1) & (yte == 1)).sum())
    fp = int(((pred == 1) & (yte == 0)).sum())
    fn = int(((pred == 0) & (yte == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("\n================= SURFACE BASELINE (non-triviality gate) =================")
    print(f"features: n_steps, ext_email_total, ext_email_any, ext_url, "
          f"sink_count, keyword_hits")
    print(f"\nAttacked-class  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")

    # per-category flag rate (the part that matters)
    flagged = defaultdict(int)
    total = defaultdict(int)
    for c, p in zip(cats_te, pred):
        total[c] += 1
        flagged[c] += int(p == 1)
    print("\nFraction flagged as ATTACKED, per true category:")
    for c in sorted(total):
        role = "ATTACK (want high)" if c in POSITIVE_CATEGORIES else "legit (want low)"
        print(f"  {c:14s}: {flagged[c]:4d}/{total[c]:<4d} "
              f"= {flagged[c]/max(total[c],1):.2f}   [{role}]")

    hn = flagged.get("hard_negative", 0) / max(total.get("hard_negative", 1), 1)
    at = flagged.get("attacked", 0) / max(total.get("attacked", 1), 1)
    print("\n--- VERDICT ---")
    print(f"hard_negative false-positive rate : {hn:.2f}")
    print(f"attacked recall                   : {at:.2f}")
    if at >= 0.85 and hn <= 0.20:
        print(">> Surface features alone separate the classes: the benchmark would be shortcut-able.")
    elif at < 0.6:
        print(">> Surface features miss many attacks: detection requires reading the sequence.")
    else:
        print(">> Mixed: surface features carry signal but fail on the hard cases.")
    print("==========================================================================")
    print("coef (standardized):",
          dict(zip(["n_steps","ext_total","ext_any","ext_url","sink","kw"],
                   [round(c, 2) for c in clf.coef_[0]])))


if __name__ == "__main__":
    main()
