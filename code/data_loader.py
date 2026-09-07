#!/usr/bin/env python3
"""
AgentDrift data loader (no framework dependencies).

    from data_loader import load_trajectories, world_known_sets, EMAIL_RE

    train = load_trajectories("data", "train")                        # stratified split (folders)
    train = load_trajectories("data", "train", split="task_disjoint") # via splits/task_disjoint.json

Each trajectory is a dict with the fields documented in README.md; every step in
traj["steps"] carries a label in {"benign", "injection_point", "hijacked",
"failed_injection"}.
"""
import os
import re
import json
import glob
from collections import Counter

VALID_CATEGORIES = {"attacked", "benign", "failed_attack", "hard_negative"}
POSITIVE_CATEGORIES = {"attacked"}          # default: only successful compromise is positive
STEP_LABELS = ["benign", "injection_point", "hijacked", "failed_injection"]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://[\w.-]+")


def _repo_root(data_root):
    return os.path.dirname(os.path.abspath(os.path.expanduser(data_root)))


def iter_all(data_root):
    """Yield every trajectory in the corpus regardless of split."""
    for part in ("train", "val", "test"):
        for path in sorted(glob.glob(os.path.join(os.path.expanduser(data_root), part, "*.json"))):
            with open(path) as f:
                yield json.load(f)


def load_trajectories(data_root="data", part="train", split="stratified"):
    """Load one part (train/val/test) of a split.

    split="stratified"    -> the folder layout data/{train,val,test}
    split="task_disjoint" -> ids taken from splits/task_disjoint.json
    """
    data_root = os.path.expanduser(data_root)
    if split == "stratified":
        trajs = []
        for path in sorted(glob.glob(os.path.join(data_root, part, "*.json"))):
            with open(path) as f:
                trajs.append(json.load(f))
    else:
        manifest = os.path.join(_repo_root(data_root), "splits", f"{split}.json")
        with open(manifest) as f:
            assignment = json.load(f)
        trajs = [t for t in iter_all(data_root) if assignment.get(t["id"]) == part]
    return [t for t in trajs if t.get("category") in VALID_CATEGORIES and t.get("steps")]


def world_known_sets(world):
    """Known-good emails and domains from the trajectory's world object."""
    emails, domains = set(), set()
    if isinstance(world, dict):
        ue = (world.get("email") or "").lower()
        if "@" in ue:
            emails.add(ue)
            domains.add(ue.split("@")[-1])
        for c in world.get("contacts") or []:
            ce = (c.get("email") or "").lower()
            if "@" in ce:
                emails.add(ce)
                domains.add(ce.split("@")[-1])
    return emails, domains


def step_world_features(step, known_emails, known_domains):
    """Four world-grounded per-step features derived from step['args']:
    [args_has_email, args_has_external_email, args_external_email_frac, args_has_external_url]."""
    try:
        blob = json.dumps(step.get("args", {}), ensure_ascii=False).lower()
    except Exception:
        blob = str(step.get("args", {})).lower()
    emails = EMAIL_RE.findall(blob)
    urls = URL_RE.findall(blob)
    ext = [e for e in emails if e.split("@")[-1] not in known_domains and e not in known_emails]
    ext_url = any(not any(d and d in u for d in known_domains) for u in urls)
    return [
        1.0 if emails else 0.0,
        1.0 if ext else 0.0,
        (len(ext) / len(emails)) if emails else 0.0,
        1.0 if ext_url else 0.0,
    ]


def render_step(step):
    """Canonical text rendering of a step for text encoders."""
    args = json.dumps(step.get("args", {}), ensure_ascii=False)
    return f"TOOL: {step.get('tool','')} | THOUGHT: {step.get('thought','')} | ARGS: {args} | OBS: {step.get('obs','')}"


def trajectory_label(traj, any_attempt=False):
    """1 if the trajectory is a successful compromise (default) or, with any_attempt=True,
    if an injection was delivered at all (attacked or failed_attack)."""
    if any_attempt:
        return int(traj.get("category") in {"attacked", "failed_attack"})
    return int(traj.get("category") in POSITIVE_CATEGORIES)


def injection_index(traj):
    """1-based index of the injection_point / failed_injection step, or None."""
    for i, s in enumerate(traj.get("steps", []), start=1):
        if s.get("label") in ("injection_point", "failed_injection"):
            return i
    return None


def anonymize_world(traj):
    """Deep copy in which person names, company names, and email addresses are replaced by
    placeholders that preserve the world's relations (USER, CONTACT_k, in-domain vs.
    out-of-domain addresses). Recommended before training text models; see README,
    'World-identity leakage'."""
    import copy
    t = copy.deepcopy(traj)
    w = t.get("world", {})
    mapping = {}
    if w.get("user"):
        mapping[w["user"]] = "USER"
    for k, c in enumerate(w.get("contacts") or [], start=1):
        if c.get("name"):
            mapping[c["name"]] = f"CONTACT_{k}"
    if w.get("company"):
        mapping[w["company"]] = "COMPANY"
    _, domains = world_known_sets(w)

    def sub_text(s):
        if not isinstance(s, str):
            return s
        for name, ph in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
            s = s.replace(name, ph)

        def email_sub(m):
            dom = m.group(0).rsplit("@", 1)[-1].lower()
            return "user@internal.example" if dom in domains else "user@external.example"
        return EMAIL_RE.sub(email_sub, s)

    def walk(x):
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return sub_text(x)

    t["task"] = sub_text(t.get("task", ""))
    t["steps"] = walk(t.get("steps", []))
    t["world"] = {"user": "USER", "email": "user@internal.example", "company": "COMPANY",
                  "date": w.get("date", ""),
                  "contacts": [{"name": f"CONTACT_{k}", "email": "user@internal.example",
                                "relation": c.get("relation", "")}
                               for k, c in enumerate(w.get("contacts") or [], start=1)]}
    return t


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--split", default="stratified", choices=["stratified", "task_disjoint"])
    args = ap.parse_args()
    for part in ("train", "val", "test"):
        ts = load_trajectories(args.root, part, split=args.split)
        print(part, len(ts), dict(Counter(t["category"] for t in ts)))
