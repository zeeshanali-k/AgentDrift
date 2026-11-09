"""Step serialization, identity-free world feature extraction, and embedding cache.

Implements Section V-A of:
'DriftNet: A Dual-Head Trajectory Transformer for Detecting and Localizing Prompt Injection in LLM Agents'
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import torch

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://[^\s\"',;)\]]+", re.IGNORECASE)

STEP_LABELS = ["benign", "injection_point", "hijacked", "failed_injection"]
LABEL_TO_IDX = {lbl: idx for idx, lbl in enumerate(STEP_LABELS)}
IDX_TO_LABEL = {idx: lbl for idx, lbl in enumerate(STEP_LABELS)}


def serialize_step(step: Dict[str, Any]) -> str:
    """Serialize a single trajectory step in the fixed paper format (Section V-A, Listing 1)."""
    tool = step.get("tool", "")
    thought = step.get("thought", "")
    args_obj = step.get("args", {})
    if isinstance(args_obj, str):
        args_str = args_obj
    else:
        args_str = json.dumps(args_obj, sort_keys=True)
    obs = step.get("obs", "")
    return f"TOOL: {tool} | THOUGHT: {thought} | ARGS: {args_str} | OBS: {obs}"


def known_world_sets(world: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    """Return known email addresses and domains from the trajectory's world context W."""
    emails: Set[str] = set()
    user_email = world.get("email", "")
    if user_email:
        emails.add(user_email.lower().replace(" ", ""))
    for contact in world.get("contacts", []):
        c_email = contact.get("email", "")
        if c_email:
            emails.add(c_email.lower().replace(" ", ""))
    domains = {e.split("@")[-1] for e in emails if "@" in e}
    return emails, domains


def extract_step_world_features(step: Dict[str, Any], world: Dict[str, Any]) -> List[float]:
    """Compute the 4 identity-free world features for a step (Section V-A):
    [args_has_email, args_has_external_email, args_external_email_frac, args_has_external_url]
    """
    known_emails, known_domains = known_world_sets(world)
    args_obj = step.get("args", {})
    args_str = args_obj if isinstance(args_obj, str) else json.dumps(args_obj, sort_keys=True)
    args_lower = args_str.lower()

    found_emails = [e.replace(" ", "") for e in EMAIL_RE.findall(args_lower)]
    ext_emails = [
        e for e in found_emails
        if e not in known_emails and e.split("@")[-1] not in known_domains
    ]

    args_has_email = 1.0 if len(found_emails) > 0 else 0.0
    args_has_external_email = 1.0 if len(ext_emails) > 0 else 0.0

    # Matches Listing 1 where 1 external email with 3 world contacts yields 0.33
    num_contacts = max(len(world.get("contacts", [])), 1)
    args_external_email_frac = min(len(ext_emails) / float(num_contacts), 1.0)

    found_urls = URL_RE.findall(args_lower)
    has_ext_url = 0.0
    for u in found_urls:
        try:
            host = (urlparse(u).hostname or "").lower()
        except Exception:
            host = ""
        if host:
            # Check if host matches or is a subdomain of any known domain
            is_internal = any(host == d or host.endswith("." + d) for d in known_domains)
            if not is_internal:
                has_ext_url = 1.0
                break

    return [
        args_has_email,
        args_has_external_email,
        round(args_external_email_frac, 4),
        has_ext_url,
    ]


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_or_load_split_cache(
    split_name: str,
    jsonl_path: str | Path,
    cache_dir: str | Path,
    encoder=None,
    device: str = "mps",
) -> Dict[str, Any]:
    """Precompute and cache frozen all-mpnet-base-v2 step embeddings + world features."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"taskdisjoint_{split_name}.pt"

    if cache_file.exists():
        print(f"[Cache] Loading cached embeddings from {cache_file}")
        return torch.load(cache_file, map_location="cpu", weights_only=False)

    records = load_jsonl(jsonl_path)
    print(f"[Cache] Encoding {len(records)} trajectories for '{split_name}' split...")

    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-mpnet-base-v2", device=device)

    all_step_texts: List[str] = []
    traj_slices: List[Tuple[int, int]] = []
    world_feats_list: List[np.ndarray] = []
    step_labels_list: List[np.ndarray] = []
    traj_labels: List[int] = []
    meta_list: List[Dict[str, Any]] = []

    cursor = 0
    for rec in records:
        steps = rec["steps"]
        world = rec.get("world", {})
        n_steps = len(steps)
        w_feats = []
        s_labels = []
        for s in steps:
            all_step_texts.append(serialize_step(s))
            w_feats.append(extract_step_world_features(s, world))
            s_labels.append(LABEL_TO_IDX[s["label"]])

        traj_slices.append((cursor, cursor + n_steps))
        cursor += n_steps
        world_feats_list.append(np.array(w_feats, dtype=np.float32))
        step_labels_list.append(np.array(s_labels, dtype=np.int64))
        traj_labels.append(1 if rec["category"] == "attacked" else 0)
        meta_list.append({
            "id": rec["id"],
            "agent": rec.get("agent", ""),
            "category": rec.get("category", ""),
            "source_category": rec.get("source_category", ""),
            "compliance": rec.get("compliance", ""),
            "attack_type": rec.get("attack_type", ""),
            "num_steps": n_steps,
        })

    embeddings = encoder.encode(
        all_step_texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    traj_embeddings = [embeddings[start:end] for (start, end) in traj_slices]

    payload = {
        "embeddings": traj_embeddings,       # List[np.ndarray of shape (T, 768)]
        "world_features": world_feats_list,  # List[np.ndarray of shape (T, 4)]
        "step_labels": step_labels_list,     # List[np.ndarray of shape (T,)]
        "traj_labels": np.array(traj_labels, dtype=np.int64),
        "meta": meta_list,
    }
    torch.save(payload, cache_file)
    print(f"[Cache] Saved {split_name} cache to {cache_file}")
    return payload
