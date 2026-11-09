"""Training and Evaluation script for DriftNet + Real-Time Streaming Prefix Benchmark.

Reproduces Algorithm 1 and Section V-VII of the DriftNet paper on `data_taskdisjoint`,
and implements the Section XI Future Work: 'Streaming operation' over growing prefixes.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .features import (
    IDX_TO_LABEL,
    STEP_LABELS,
    build_or_load_split_cache,
)
from .model import DriftNet, count_trainable_parameters


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrajectoryDataset(Dataset):
    """Dataset of cached trajectory step embeddings and world features."""

    def __init__(self, cache_data: Dict[str, Any]):
        self.embeddings = cache_data["embeddings"]
        self.world_features = cache_data["world_features"]
        self.step_labels = cache_data["step_labels"]
        self.traj_labels = cache_data["traj_labels"]
        self.meta = cache_data["meta"]

    def __len__(self) -> int:
        return len(self.traj_labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, int]:
        emb = self.embeddings[idx]
        wf = self.world_features[idx]
        x = np.concatenate([emb, wf], axis=-1).astype(np.float32)  # (T, 772)
        z = self.step_labels[idx]  # (T,)
        y = int(self.traj_labels[idx])
        return x, z, y


def collate_trajectories(batch: List[Tuple[np.ndarray, np.ndarray, int]]):
    B = len(batch)
    max_t = max(item[0].shape[0] for item in batch)
    feat_dim = batch[0][0].shape[1]

    x_pad = np.zeros((B, max_t, feat_dim), dtype=np.float32)
    z_pad = np.full((B, max_t), fill_value=-100, dtype=np.int64)
    mask = np.zeros((B, max_t), dtype=bool)
    y_arr = np.zeros((B,), dtype=np.float32)

    for i, (x, z, y) in enumerate(batch):
        t = x.shape[0]
        x_pad[i, :t, :] = x
        z_pad[i, :t] = z
        mask[i, :t] = True
        y_arr[i] = float(y)

    return (
        torch.from_numpy(x_pad),
        torch.from_numpy(mask),
        torch.from_numpy(y_arr),
        torch.from_numpy(z_pad),
    )


def compute_joint_loss(
    traj_logits: torch.Tensor,
    step_logits: torch.Tensor,
    y_true: torch.Tensor,
    z_true: torch.Tensor,
    pos_weight: torch.Tensor,
    step_weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute class-weighted joint loss L = L_traj + L_step (Eq. 12-15 of paper)."""
    # Eq. 13: Class-weighted binary cross entropy with w_+ = 1.27
    probs = torch.sigmoid(traj_logits).clamp(min=1e-7, max=1.0 - 1e-7)
    loss_traj = -torch.mean(
        pos_weight * y_true * torch.log(probs) + (1.0 - y_true) * torch.log(1.0 - probs)
    )

    # Eq. 14-15: Class-weighted categorical cross entropy over valid steps
    # Normalized by sum of active weights: sum(m_{i,t} * alpha_{z_{i,t}})
    B, T, C = step_logits.shape
    loss_step = F.cross_entropy(
        step_logits.view(B * T, C),
        z_true.view(B * T),
        weight=step_weights,
        ignore_index=-100,
        reduction="mean",
    )
    return loss_traj + loss_step, loss_traj, loss_step


@torch.no_grad()
def evaluate_full_trajectories(
    model: DriftNet,
    cache_data: Dict[str, Any],
    device: str,
    pos_weight: torch.Tensor,
    step_weights: torch.Tensor,
) -> Dict[str, Any]:
    """Evaluate DriftNet on completed trajectories (Section III-D & Section VII)."""
    model.eval()
    dataset = TrajectoryDataset(cache_data)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_trajectories)

    all_y_true: List[int] = []
    all_y_prob: List[float] = []
    all_z_true: List[np.ndarray] = []
    all_z_pred: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for x, mask, y, z in loader:
        x_dev = x.to(device)
        mask_dev = mask.to(device)
        y_dev = y.to(device)
        z_dev = z.to(device)

        traj_logits, step_logits = model(x_dev, mask_dev)
        loss, _, _ = compute_joint_loss(traj_logits, step_logits, y_dev, z_dev, pos_weight, step_weights)
        total_loss += float(loss.item())
        n_batches += 1

        probs = torch.sigmoid(traj_logits).cpu().numpy()
        step_preds = step_logits.argmax(dim=-1).cpu().numpy()
        z_np = z.numpy()
        mask_np = mask.numpy()

        for i in range(len(probs)):
            all_y_true.append(int(y[i].item()))
            all_y_prob.append(float(probs[i]))
            valid_len = int(mask_np[i].sum())
            all_z_true.append(z_np[i, :valid_len])
            all_z_pred.append(step_preds[i, :valid_len])

    y_true_arr = np.array(all_y_true)
    y_prob_arr = np.array(all_y_prob)
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.int64)

    tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
    tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    # Category flag rates (Table VI)
    meta = cache_data["meta"]
    cat_counts: Dict[str, int] = {}
    cat_flagged: Dict[str, int] = {}
    pattern_counts: Dict[str, int] = {}
    pattern_recalled: Dict[str, int] = {}
    domain_stats: Dict[str, Dict[str, int]] = {}

    # Localization metrics on attacked trajectories (Eq. 4 & 5)
    em_i_all: List[float] = []
    iou_h_all: List[float] = []
    pattern_loc: Dict[str, Dict[str, List[float]]] = {
        "full_hijack": {"em": [], "iou": []},
        "partial_hijack": {"em": [], "iou": []},
        "delayed_execution": {"em": [], "iou": []},
    }

    for i, m in enumerate(meta):
        cat = m["category"]
        dom = m["agent"]
        pred = int(y_pred_arr[i])
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        cat_flagged[cat] = cat_flagged.get(cat, 0) + pred

        if dom not in domain_stats:
            domain_stats[dom] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "n": 0}
        domain_stats[dom]["n"] += 1
        if y_true_arr[i] == 1 and pred == 1:
            domain_stats[dom]["tp"] += 1
        elif y_true_arr[i] == 0 and pred == 1:
            domain_stats[dom]["fp"] += 1
        elif y_true_arr[i] == 1 and pred == 0:
            domain_stats[dom]["fn"] += 1
        else:
            domain_stats[dom]["tn"] += 1

        if cat == "attacked":
            comp = m["compliance"]
            pattern_counts[comp] = pattern_counts.get(comp, 0) + 1
            pattern_recalled[comp] = pattern_recalled.get(comp, 0) + pred

            z_gold = all_z_true[i]
            z_hat = all_z_pred[i]
            gold_I = set(np.where(z_gold == 1)[0].tolist())
            pred_I = set(np.where(z_hat == 1)[0].tolist())
            em_val = 1.0 if gold_I == pred_I else 0.0

            gold_H = set(np.where(z_gold == 2)[0].tolist())
            pred_H = set(np.where(z_hat == 2)[0].tolist())
            union_H = gold_H | pred_H
            iou_val = 1.0 if len(union_H) == 0 else len(gold_H & pred_H) / float(len(union_H))

            em_i_all.append(em_val)
            iou_h_all.append(iou_val)
            if comp in pattern_loc:
                pattern_loc[comp]["em"].append(em_val)
                pattern_loc[comp]["iou"].append(iou_val)

    # Step-level metrics (Table VII)
    flat_z_true = np.concatenate(all_z_true)
    flat_z_pred = np.concatenate(all_z_pred)
    step_metrics = {}
    for c_idx, c_name in enumerate(STEP_LABELS):
        s_tp = int(((flat_z_pred == c_idx) & (flat_z_true == c_idx)).sum())
        s_fp = int(((flat_z_pred == c_idx) & (flat_z_true != c_idx)).sum())
        s_fn = int(((flat_z_pred != c_idx) & (flat_z_true == c_idx)).sum())
        s_p = s_tp / max(s_tp + s_fp, 1)
        s_r = s_tp / max(s_tp + s_fn, 1)
        s_f1 = 2 * s_p * s_r / max(s_p + s_r, 1e-12)
        step_metrics[c_name] = {
            "precision": round(s_p, 4),
            "recall": round(s_r, 4),
            "f1": round(s_f1, 4),
            "support": int((flat_z_true == c_idx).sum()),
        }

    domain_summary = {}
    for dom, ds in sorted(domain_stats.items()):
        d_rec = ds["tp"] / max(ds["tp"] + ds["fn"], 1)
        d_fa = ds["fp"] / max(ds["fp"] + ds["tn"], 1)
        d_p = ds["tp"] / max(ds["tp"] + ds["fp"], 1)
        d_f1 = 2 * d_p * d_rec / max(d_p + d_rec, 1e-12)
        domain_summary[dom] = {
            "n": ds["n"],
            "recall": round(d_rec, 4),
            "false_alarm_rate": round(d_fa, 4),
            "f1": round(d_f1, 4),
        }

    return {
        "loss": round(total_loss / max(n_batches, 1), 4),
        "trajectory": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "category_flag_rates": {
            cat: {
                "flagged": cat_flagged.get(cat, 0),
                "total": cat_counts.get(cat, 0),
                "rate": round(cat_flagged.get(cat, 0) / max(cat_counts.get(cat, 1), 1), 4),
            }
            for cat in ["attacked", "benign", "failed_attack", "hard_negative"]
        },
        "pattern_recall": {
            comp: {
                "recalled": pattern_recalled.get(comp, 0),
                "total": pattern_counts.get(comp, 0),
                "rate": round(pattern_recalled.get(comp, 0) / max(pattern_counts.get(comp, 1), 1), 4),
            }
            for comp in ["full_hijack", "partial_hijack", "delayed_execution"]
        },
        "localization": {
            "overall_em_i": round(float(np.mean(em_i_all)), 4),
            "overall_iou_h": round(float(np.mean(iou_h_all)), 4),
            "by_pattern": {
                comp: {
                    "n": len(vals["em"]),
                    "em_i": round(float(np.mean(vals["em"])), 4) if vals["em"] else 0.0,
                    "iou_h": round(float(np.mean(vals["iou"])), 4) if vals["iou"] else 0.0,
                }
                for comp, vals in pattern_loc.items()
            },
        },
        "step_metrics": step_metrics,
        "domain_summary": domain_summary,
    }


@torch.no_grad()
def evaluate_streaming_prefixes(
    model: DriftNet,
    cache_data: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """Evaluate DriftNet in Real-Time Streaming Mode across growing prefixes x_{1:1}..x_{1:T}.

    Directly implements Section XI Future Work ('Streaming operation'):
    Measures:
    1. Exact First-Hijack Step Detection Rate (caught at the exact step the first H action occurs)
    2. Retroactive Entry-Point (I) Pinpointing Rate at the moment of First Hijack
    3. Mean Detection Delay (steps after first hijacked action)
    4. Downstream Steps Saved / Prevented by Halting at First Hijack Detection
    5. Online Trajectory Detection Recall & False-Alarm Rates across compliance patterns
    """
    model.eval()
    embeddings = cache_data["embeddings"]
    world_features = cache_data["world_features"]
    step_labels = cache_data["step_labels"]
    meta = cache_data["meta"]

    total_attacked = 0
    caught_ever = 0
    caught_at_first_hijack = 0
    caught_within_1_step = 0
    exact_entry_point_at_halt = 0
    detection_delays: List[int] = []
    steps_saved_list: List[int] = []
    hijacked_actions_prevented_list: List[int] = []

    pattern_streaming: Dict[str, Dict[str, Any]] = {
        "full_hijack": {"n": 0, "caught_at_first_h": 0, "caught_ever": 0, "entry_exact_at_halt": 0, "steps_saved": []},
        "partial_hijack": {"n": 0, "caught_at_first_h": 0, "caught_ever": 0, "entry_exact_at_halt": 0, "steps_saved": []},
        "delayed_execution": {"n": 0, "caught_at_first_h": 0, "caught_ever": 0, "entry_exact_at_halt": 0, "steps_saved": []},
    }

    neg_categories = {
        "benign": {"total": 0, "false_halt": 0},
        "failed_attack": {"total": 0, "false_halt": 0},
        "hard_negative": {"total": 0, "false_halt": 0},
    }

    # Prefix length curve (prefix relative to first H step: -1, 0, +1, +2)
    rel_step_hits = {-1: [], 0: [], 1: [], 2: []}

    for i, m in enumerate(meta):
        emb = embeddings[i]
        wf = world_features[i]
        z_gold = step_labels[i]
        T = len(z_gold)
        cat = m["category"]

        snapshots = model.predict_prefix_stream(emb, wf, device=device)

        # Online stopping rule (Section XI): halt as soon as Head 1 flags compromise (>=0.5)
        # OR Head 2 labels a step as hijacked ('H')
        halt_step = None
        for t_idx, snap in enumerate(snapshots):
            if snap["traj_compromised"]:
                halt_step = t_idx
                break

        if cat == "attacked":
            total_attacked += 1
            comp = m["compliance"]
            if comp in pattern_streaming:
                pattern_streaming[comp]["n"] += 1

            gold_h_indices = np.where(z_gold == 2)[0]
            gold_i_indices = set(np.where(z_gold == 1)[0].tolist())
            first_h = int(gold_h_indices[0]) if len(gold_h_indices) > 0 else 0

            # Record relative prefix detection rates around first_h
            for rel_offset in [-1, 0, 1, 2]:
                target_t = first_h + rel_offset
                if 0 <= target_t < T:
                    rel_step_hits[rel_offset].append(1.0 if snapshots[target_t]["traj_compromised"] else 0.0)

            if halt_step is not None and halt_step >= first_h:
                caught_ever += 1
                delay = halt_step - first_h
                detection_delays.append(delay)
                if comp in pattern_streaming:
                    pattern_streaming[comp]["caught_ever"] += 1

                if delay == 0:
                    caught_at_first_hijack += 1
                    if comp in pattern_streaming:
                        pattern_streaming[comp]["caught_at_first_h"] += 1
                if delay <= 1:
                    caught_within_1_step += 1

                # Check if injection_point (I) is accurately localized at the exact halt step!
                pred_i_at_halt = set(snapshots[halt_step]["injection_points"])
                if pred_i_at_halt == gold_i_indices:
                    exact_entry_point_at_halt += 1
                    if comp in pattern_streaming:
                        pattern_streaming[comp]["entry_exact_at_halt"] += 1

                # Downstream steps saved after halt_step
                saved = (T - 1) - halt_step
                steps_saved_list.append(saved)
                if comp in pattern_streaming:
                    pattern_streaming[comp]["steps_saved"].append(saved)

                # How many gold H steps from halt_step onwards were intercepted/prevented
                prevented_h = int((gold_h_indices >= halt_step).sum())
                hijacked_actions_prevented_list.append(prevented_h)
            elif halt_step is not None and halt_step < first_h:
                # Fired at the injection point before hijacked action even executed!
                caught_ever += 1
                caught_at_first_hijack += 1
                caught_within_1_step += 1
                detection_delays.append(0)
                if comp in pattern_streaming:
                    pattern_streaming[comp]["caught_ever"] += 1
                    pattern_streaming[comp]["caught_at_first_h"] += 1
                saved = (T - 1) - halt_step
                steps_saved_list.append(saved)
                if comp in pattern_streaming:
                    pattern_streaming[comp]["steps_saved"].append(saved)
                hijacked_actions_prevented_list.append(len(gold_h_indices))
        else:
            if cat in neg_categories:
                neg_categories[cat]["total"] += 1
                if halt_step is not None:
                    neg_categories[cat]["false_halt"] += 1

    pattern_summary = {}
    for comp, ps in pattern_streaming.items():
        n = max(ps["n"], 1)
        pattern_summary[comp] = {
            "n": ps["n"],
            "instant_first_hijack_recall": round(ps["caught_at_first_h"] / n, 4),
            "online_overall_recall": round(ps["caught_ever"] / n, 4),
            "retroactive_entry_exact_match": round(ps["entry_exact_at_halt"] / max(ps["caught_ever"], 1), 4),
            "mean_downstream_steps_saved": round(float(np.mean(ps["steps_saved"])) if ps["steps_saved"] else 0.0, 2),
        }

    return {
        "total_attacked_trajectories": total_attacked,
        "online_overall_recall": round(caught_ever / max(total_attacked, 1), 4),
        "instant_first_hijack_recall": round(caught_at_first_hijack / max(total_attacked, 1), 4),
        "within_1_step_recall": round(caught_within_1_step / max(total_attacked, 1), 4),
        "retroactive_entry_point_em_at_halt": round(exact_entry_point_at_halt / max(caught_ever, 1), 4),
        "mean_detection_delay_steps": round(float(np.mean(detection_delays)) if detection_delays else 0.0, 3),
        "mean_downstream_steps_saved": round(float(np.mean(steps_saved_list)) if steps_saved_list else 0.0, 2),
        "total_hijacked_actions_intercepted": int(sum(hijacked_actions_prevented_list)),
        "total_downstream_steps_saved": int(sum(steps_saved_list)),
        "relative_step_detection_curve": {
            "at_injection_point_before_hijack (t_H - 1)": round(float(np.mean(rel_step_hits[-1])), 4),
            "at_first_hijacked_action (t_H)": round(float(np.mean(rel_step_hits[0])), 4),
            "one_step_after_first_hijack (t_H + 1)": round(float(np.mean(rel_step_hits[1])), 4),
            "two_steps_after_first_hijack (t_H + 2)": round(float(np.mean(rel_step_hits[2])), 4),
        },
        "by_pattern": pattern_summary,
        "negative_category_false_halt_rates": {
            cat: {
                "false_halts": d["false_halt"],
                "total": d["total"],
                "rate": round(d["false_halt"] / max(d["total"], 1), 4),
            }
            for cat, d in neg_categories.items()
        },
    }


def main():
    set_seed(42)
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data_taskdisjoint"
    cache_dir = repo_root / "driftnet" / "cache"
    artifacts_dir = repo_root / "driftnet" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DriftNet] Using compute device: {device}")

    # 1. Build or load cached embeddings for train, val, test
    from sentence_transformers import SentenceTransformer
    encoder = None
    for split in ["train", "val", "test"]:
        cache_file = cache_dir / f"taskdisjoint_{split}.pt"
        if not cache_file.exists() and encoder is None:
            print("[DriftNet] Loading frozen sentence encoder 'all-mpnet-base-v2'...")
            encoder = SentenceTransformer("all-mpnet-base-v2", device=device)

    train_cache = build_or_load_split_cache("train", data_dir / "train.jsonl", cache_dir, encoder, device)
    val_cache = build_or_load_split_cache("val", data_dir / "val.jsonl", cache_dir, encoder, device)
    test_cache = build_or_load_split_cache("test", data_dir / "test.jsonl", cache_dir, encoder, device)

    # Free sentence encoder memory during training
    del encoder

    # 2. Initialize DriftNet with sweep-selected configuration (Table IV: L=3, d=256, dropout=0.3, lr=1e-3, wd=0.05)
    model = DriftNet(
        input_dim=772,
        d_model=256,
        nhead=4,
        num_layers=3,
        dim_feedforward=512,
        dropout=0.3,
        num_step_classes=4,
    ).to(device)

    n_params = count_trainable_parameters(model)
    print(f"[DriftNet] Model initialized with {n_params:,} trainable parameters (Paper L=3 target: 1,845,637)")

    ckpt_path = artifacts_dir / "driftnet_best.pt"
    pos_weight = torch.tensor(1.27, dtype=torch.float32, device=device)
    step_weights = torch.tensor([0.08, 0.76, 0.34, 2.81], dtype=torch.float32, device=device)

    train_history = []
    if ckpt_path.exists():
        print(f"[DriftNet] Loading existing best checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        train_history = ckpt.get("train_history", [])
    else:
        train_dataset = TrajectoryDataset(train_cache)
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_trajectories,
        )

        max_epochs = 40
        total_steps = max_epochs * len(train_loader)
        warmup_steps = int(0.1 * total_steps)
        peak_lr = 1e-3

        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.05)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_f1 = -1.0
        patience = 8
        patience_counter = 0
        best_epoch = 0

        print(f"[DriftNet] Starting training (up to {max_epochs} epochs, patience={patience})...")
        t0 = time.time()
        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            for x, mask, y, z in train_loader:
                x_dev = x.to(device)
                mask_dev = mask.to(device)
                y_dev = y.to(device)
                z_dev = z.to(device)

                optimizer.zero_grad()
                traj_logits, step_logits = model(x_dev, mask_dev)
                loss, _, _ = compute_joint_loss(
                    traj_logits, step_logits, y_dev, z_dev, pos_weight, step_weights
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                epoch_loss += float(loss.item())

            avg_train_loss = epoch_loss / len(train_loader)
            val_metrics = evaluate_full_trajectories(model, val_cache, device, pos_weight, step_weights)
            val_f1 = val_metrics["trajectory"]["f1"]
            val_loss = val_metrics["loss"]

            train_history.append({
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_em_i": val_metrics["localization"]["overall_em_i"],
                "val_iou_h": val_metrics["localization"]["overall_iou_h"],
            })

            print(
                f"  Epoch {epoch:02d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | "
                f"Val EM_I: {val_metrics['localization']['overall_em_i']:.4f} | "
                f"Val IoU_H: {val_metrics['localization']['overall_iou_h']:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "val_f1": best_val_f1,
                        "trainable_parameters": n_params,
                        "model_state_dict": model.state_dict(),
                        "train_history": train_history,
                    },
                    ckpt_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"[DriftNet] Early stopping triggered at epoch {epoch} (best epoch: {best_epoch}, Val F1: {best_val_f1:.4f})")
                    break

        elapsed = time.time() - t0
        print(f"[DriftNet] Training complete in {elapsed:.1f}s. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

    # 3. Single Held-Out Test Evaluation (Full Trajectory + Real-Time Streaming Prefixes)
    print("\n[DriftNet] Evaluating on Held-Out Task-Disjoint Test Set (1,722 trajectories)...")
    val_full = evaluate_full_trajectories(model, val_cache, device, pos_weight, step_weights)
    test_full = evaluate_full_trajectories(model, test_cache, device, pos_weight, step_weights)
    print(
        f"  [Full Trajectory Test] Precision: {test_full['trajectory']['precision']:.4f} | "
        f"Recall: {test_full['trajectory']['recall']:.4f} | "
        f"F1: {test_full['trajectory']['f1']:.4f} | "
        f"EM_I: {test_full['localization']['overall_em_i']:.4f} | "
        f"IoU_H: {test_full['localization']['overall_iou_h']:.4f}"
    )

    print("[DriftNet] Running Real-Time Streaming Prefix Evaluation on Held-Out Test Set...")
    test_streaming = evaluate_streaming_prefixes(model, test_cache, device)
    print(
        f"  [Streaming Prefix Test] Instant First-Hijack Recall: {test_streaming['instant_first_hijack_recall']:.4f} | "
        f"Online Overall Recall: {test_streaming['online_overall_recall']:.4f} | "
        f"Retroactive Entry-Point Exact Match at Halt: {test_streaming['retroactive_entry_point_em_at_halt']:.4f} | "
        f"Downstream Steps Saved: {test_streaming['total_downstream_steps_saved']}"
    )

    summary_payload = {
        "model_info": {
            "name": "DriftNet (Dual-Head Trajectory Transformer)",
            "trainable_parameters": n_params,
            "encoder": "all-mpnet-base-v2 (frozen, 768-d)",
            "world_features": [
                "args_has_email",
                "args_has_external_email",
                "args_external_email_frac",
                "args_has_external_url",
            ],
            "layers": 3,
            "d_model": 256,
            "nhead": 4,
            "ffn_dim": 512,
        },
        "validation_full": val_full,
        "test_full": test_full,
        "test_streaming": test_streaming,
        "train_history": train_history,
    }

    eval_json_path = artifacts_dir / "eval_results.json"
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[DriftNet] Saved full + streaming evaluation report to {eval_json_path}")


if __name__ == "__main__":
    main()
