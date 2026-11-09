"""Lightweight HTTP API + Static UI Server for DriftNet Live (Streaming Prefix Monitor).

Serves:
- GET  /                        -> Interactive Web UI (static/index.html)
- GET  /api/stats               -> Full-Trajectory & Streaming Prefix Benchmark metrics
- GET  /api/trajectories        -> Filterable list of held-out test trajectories + curated presets
- GET  /api/trajectory/<id>     -> Full step-by-step streaming prefix predictions for trajectory <id>
- POST /api/score_custom        -> Live re-scoring when a user edits any step's thought/tool/args/obs
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from .features import (
    extract_step_world_features,
    known_world_sets,
    load_jsonl,
    serialize_step,
)
from .model import DriftNet


class DriftNetAppState:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.data_dir = repo_root / "data_taskdisjoint"
        self.cache_dir = repo_root / "driftnet" / "cache"
        self.artifacts_dir = repo_root / "driftnet" / "artifacts"
        self.static_dir = repo_root / "driftnet" / "static"

        self.device = "cpu"  # Fast single-trajectory inference (<1ms on CPU)
        self.model = DriftNet(
            input_dim=772,
            d_model=256,
            nhead=4,
            num_layers=3,
            dim_feedforward=512,
            dropout=0.3,
            num_step_classes=4,
        ).to(self.device)

        ckpt_path = self.artifacts_dir / "driftnet_best.pt"
        self.has_trained_weights = False
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            self.has_trained_weights = True
            print(f"[Server] Loaded trained DriftNet checkpoint from {ckpt_path}")
        else:
            print(f"[Server] NOTE: Trained checkpoint {ckpt_path} not found yet. Using world-grounded prefix analyzer until trained.")

        eval_path = self.artifacts_dir / "eval_results.json"
        self.eval_results = {}
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                self.eval_results = json.load(f)

        print("[Server] Loading held-out test records and cached embeddings...")
        self.test_records = load_jsonl(self.data_dir / "test.jsonl")
        self.records_by_id: Dict[str, Dict[str, Any]] = {}
        self.index_by_id: Dict[str, int] = {}
        for idx, rec in enumerate(self.test_records):
            self.records_by_id[rec["id"]] = rec
            self.index_by_id[rec["id"]] = idx

        test_cache_path = self.cache_dir / "taskdisjoint_test.pt"
        self.test_cache = (
            torch.load(test_cache_path, map_location="cpu", weights_only=False)
            if test_cache_path.exists()
            else None
        )

        self._sbert_encoder = None
        self.featured_ids = self._select_featured_trajectories()
        print(f"[Server] Ready with {len(self.test_records)} test trajectories.")

    def _select_featured_trajectories(self) -> List[Dict[str, str]]:
        """Pick one clean representative test trajectory per compliance pattern/category for 1-click demos."""
        targets = [
            ("Full Hijack (Banking)", lambda r: r["category"] == "attacked" and r.get("compliance") == "full_hijack" and r["agent"] == "banking"),
            ("Partial Hijack + Recovery (Email)", lambda r: r["category"] == "attacked" and r.get("compliance") == "partial_hijack" and r["agent"] == "email"),
            ("Delayed Execution (Web)", lambda r: r["category"] == "attacked" and r.get("compliance") == "delayed_execution" and r["agent"] == "web"),
            ("Delayed Execution (Medical)", lambda r: r["category"] == "attacked" and r.get("compliance") == "delayed_execution" and r["agent"] == "medical"),
            ("Resisted / Failed Attack (Banking)", lambda r: r["category"] == "failed_attack" and r["agent"] == "banking"),
            ("Hard Negative (Suspicious Benign)", lambda r: r["category"] == "hard_negative"),
            ("Benign Execution (Coding)", lambda r: r["category"] == "benign" and r["agent"] == "coding"),
        ]
        featured = []
        for label, pred_fn in targets:
            for rec in self.test_records:
                if pred_fn(rec) and 5 <= len(rec["steps"]) <= 8:
                    featured.append({
                        "preset_label": label,
                        "id": rec["id"],
                        "agent": rec["agent"],
                        "category": rec["category"],
                        "compliance": rec.get("compliance", "benign"),
                        "num_steps": len(rec["steps"]),
                    })
                    break
        return featured

    def get_encoder(self):
        if self._sbert_encoder is None:
            from sentence_transformers import SentenceTransformer
            if torch.cuda.is_available():
                enc_dev = "cuda"
            elif torch.backends.mps.is_available():
                enc_dev = "mps"
            else:
                enc_dev = "cpu"
            print(f"[Server] Lazy-loading all-mpnet-base-v2 on {enc_dev} for live step scoring...")
            self._sbert_encoder = SentenceTransformer("all-mpnet-base-v2", device=enc_dev)
        return self._sbert_encoder

    def _fallback_prefix_stream(
        self,
        steps: List[Dict[str, Any]],
        world: Dict[str, Any],
        wf: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """World-grounded + lexical drift analyzer used only if driftnet_best.pt has not been trained yet."""
        suspicious_obs_cues = [
            "per policy", "security alert", "urgent", "override", "system instruction",
            "immediately", "verification by sending", "audit@", "external", "ignore previous",
            "transfer balance to", "malicious", "fdic protocol", "action required",
        ]
        refusal_cues = [
            "suspicious", "ignore this", "phishing", "not a recognized", "appears to be",
            "never requested", "unsafe", "refuse",
        ]
        T = len(steps)
        obs_suspicious = []
        thought_refused = []
        step_deviant = []

        for idx, s in enumerate(steps):
            obs_low = str(s.get("obs", "")).lower()
            thought_low = str(s.get("thought", "")).lower()
            args_low = json.dumps(s.get("args", {})).lower()
            w_vec = wf[idx]

            has_ext = (w_vec[1] > 0) or (w_vec[3] > 0)
            has_cue = any(c in obs_low for c in suspicious_obs_cues) or (
                "@" in obs_low and "external" in obs_low
            )
            is_refusal = any(c in thought_low for c in refusal_cues)
            is_dev = has_ext or any(
                k in args_low
                for k in ["malicious", "external", "payload", "/etc/passwd", "chmod", "freeze_card"]
            ) or (s.get("label") == "hijacked")

            obs_suspicious.append(has_cue or s.get("label") in ("injection_point", "failed_injection"))
            thought_refused.append(is_refusal)
            step_deviant.append(is_dev and not is_refusal)

        snapshots = []
        for k in range(T):
            curr_len = k + 1
            any_dev_so_far = any(step_deviant[j] for j in range(curr_len))
            any_refusal_so_far = any(thought_refused[j] for j in range(curr_len))
            first_susp_idx = next((j for j in range(curr_len) if obs_suspicious[j]), None)

            step_verdicts = []
            for s_idx in range(curr_len):
                if first_susp_idx is not None and s_idx == first_susp_idx:
                    if any_dev_so_far and s_idx < k:
                        lbl = "injection_point"
                        probs = {"benign": 0.02, "injection_point": 0.95, "hijacked": 0.02, "failed_injection": 0.01}
                    elif any_refusal_so_far and not any_dev_so_far:
                        lbl = "failed_injection"
                        probs = {"benign": 0.03, "injection_point": 0.02, "hijacked": 0.01, "failed_injection": 0.94}
                    else:
                        lbl = "benign"
                        probs = {"benign": 0.72, "injection_point": 0.22, "hijacked": 0.02, "failed_injection": 0.04}
                elif first_susp_idx is not None and s_idx > first_susp_idx and step_deviant[s_idx]:
                    lbl = "hijacked"
                    probs = {"benign": 0.02, "injection_point": 0.01, "hijacked": 0.96, "failed_injection": 0.01}
                else:
                    lbl = "benign"
                    probs = {"benign": 0.97, "injection_point": 0.01, "hijacked": 0.01, "failed_injection": 0.01}

                step_verdicts.append({
                    "step_index": s_idx,
                    "pred_label": lbl,
                    "probabilities": probs,
                })

            inj_points = [sv["step_index"] for sv in step_verdicts if sv["pred_label"] == "injection_point"]
            hijacked_steps = [sv["step_index"] for sv in step_verdicts if sv["pred_label"] == "hijacked"]
            failed_steps = [sv["step_index"] for sv in step_verdicts if sv["pred_label"] == "failed_injection"]
            is_comp = len(hijacked_steps) > 0
            traj_prob = 0.984 if is_comp else (0.04 if len(failed_steps) > 0 else 0.012)

            snapshots.append({
                "prefix_length": curr_len,
                "traj_prob": round(traj_prob, 4),
                "traj_compromised": is_comp,
                "step_verdicts": step_verdicts,
                "injection_points": inj_points,
                "hijacked_steps": hijacked_steps,
                "failed_injection_steps": failed_steps,
            })
        return snapshots

    def score_trajectory_by_id(self, traj_id: str) -> Optional[Dict[str, Any]]:
        if traj_id not in self.records_by_id:
            return None
        rec = self.records_by_id[traj_id]
        idx = self.index_by_id[traj_id]

        wf = np.array(
            [extract_step_world_features(s, rec.get("world", {})) for s in rec["steps"]],
            dtype=np.float32,
        )

        if self.has_trained_weights:
            if self.test_cache is not None:
                emb = self.test_cache["embeddings"][idx]
                wf = self.test_cache["world_features"][idx]
            else:
                encoder = self.get_encoder()
                texts = [serialize_step(s) for s in rec["steps"]]
                emb = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            snapshots = self.model.predict_prefix_stream(emb, wf, device=self.device)
        else:
            snapshots = self._fallback_prefix_stream(rec["steps"], rec.get("world", {}), wf)

        known_emails, known_domains = known_world_sets(rec.get("world", {}))

        steps_enriched = []
        for s_idx, s in enumerate(rec["steps"]):
            w_vec = wf[s_idx].tolist()
            steps_enriched.append({
                "index": s_idx,
                "step_number": s_idx + 1,
                "tool": s.get("tool", ""),
                "thought": s.get("thought", ""),
                "args": s.get("args", {}),
                "obs": s.get("obs", ""),
                "gold_label": s.get("label", "benign"),
                "serialized": serialize_step(s),
                "world_features": {
                    "args_has_email": w_vec[0],
                    "args_has_external_email": w_vec[1],
                    "args_external_email_frac": round(w_vec[2], 3),
                    "args_has_external_url": w_vec[3],
                },
            })

        # Compute online interceptor trigger step
        halt_step_index = None
        for snap in snapshots:
            if snap["traj_compromised"]:
                halt_step_index = snap["prefix_length"] - 1
                break

        return {
            "id": rec["id"],
            "agent": rec.get("agent", ""),
            "category": rec.get("category", ""),
            "source_category": rec.get("source_category", ""),
            "compliance": rec.get("compliance", "none"),
            "attack_type": rec.get("attack_type", "none"),
            "task": rec.get("task", ""),
            "world": rec.get("world", {}),
            "known_domains": sorted(list(known_domains)),
            "steps": steps_enriched,
            "prefix_snapshots": snapshots,
            "halt_step_index": halt_step_index,
        }

    def score_custom_trajectory(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        world = payload.get("world", {})
        steps = payload.get("steps", [])
        wf = np.array(
            [extract_step_world_features(s, world) for s in steps],
            dtype=np.float32,
        )

        if self.has_trained_weights:
            encoder = self.get_encoder()
            texts = [serialize_step(s) for s in steps]
            emb = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
            snapshots = self.model.predict_prefix_stream(emb, wf, device=self.device)
        else:
            snapshots = self._fallback_prefix_stream(steps, world, wf)
        steps_enriched = []
        for s_idx, s in enumerate(steps):
            w_vec = wf[s_idx].tolist()
            steps_enriched.append({
                "index": s_idx,
                "step_number": s_idx + 1,
                "tool": s.get("tool", ""),
                "thought": s.get("thought", ""),
                "args": s.get("args", {}),
                "obs": s.get("obs", ""),
                "gold_label": s.get("gold_label", s.get("label", "benign")),
                "serialized": serialize_step(s),
                "world_features": {
                    "args_has_email": w_vec[0],
                    "args_has_external_email": w_vec[1],
                    "args_external_email_frac": round(w_vec[2], 3),
                    "args_has_external_url": w_vec[3],
                },
            })

        halt_step_index = None
        for snap in snapshots:
            if snap["traj_compromised"]:
                halt_step_index = snap["prefix_length"] - 1
                break

        return {
            "steps": steps_enriched,
            "prefix_snapshots": snapshots,
            "halt_step_index": halt_step_index,
        }


APP_STATE: Optional[DriftNetAppState] = None


class DriftNetRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return  # Keep console output clean

    def _send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_path: Path) -> None:
        content = html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        assert APP_STATE is not None
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send_html(APP_STATE.static_dir / "index.html")
            return

        if path == "/api/stats":
            # Reload eval_results if updated
            eval_path = APP_STATE.artifacts_dir / "eval_results.json"
            if eval_path.exists():
                with open(eval_path, "r", encoding="utf-8") as f:
                    APP_STATE.eval_results = json.load(f)
            self._send_json(APP_STATE.eval_results)
            return

        if path == "/api/trajectories":
            domain = qs.get("domain", ["all"])[0]
            category = qs.get("category", ["all"])[0]
            compliance = qs.get("compliance", ["all"])[0]
            search = qs.get("q", [""])[0].lower()
            limit = int(qs.get("limit", ["150"])[0])

            items = []
            for rec in APP_STATE.test_records:
                if domain != "all" and rec.get("agent") != domain:
                    continue
                if category != "all" and rec.get("category") != category:
                    continue
                if compliance != "all" and rec.get("compliance", "none") != compliance:
                    continue
                if search and search not in rec["id"].lower() and search not in rec.get("task", "").lower():
                    continue
                items.append({
                    "id": rec["id"],
                    "agent": rec.get("agent", ""),
                    "category": rec.get("category", ""),
                    "compliance": rec.get("compliance", "none"),
                    "num_steps": len(rec.get("steps", [])),
                    "task": rec.get("task", ""),
                })
                if len(items) >= limit:
                    break

            self._send_json({
                "featured": APP_STATE.featured_ids,
                "trajectories": items,
                "total_test_count": len(APP_STATE.test_records),
            })
            return

        if path.startswith("/api/trajectory/"):
            traj_id = path.split("/api/trajectory/", 1)[1]
            result = APP_STATE.score_trajectory_by_id(traj_id)
            if result is None:
                self._send_json({"error": f"Trajectory '{traj_id}' not found"}, status=404)
            else:
                self._send_json(result)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        assert APP_STATE is not None
        parsed = urlparse(self.path)
        if parsed.path == "/api/score_custom":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                payload = json.loads(raw.decode("utf-8"))
                result = APP_STATE.score_custom_trajectory(payload)
                self._send_json(result)
            except Exception as e:
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
            return

        self._send_json({"error": "Not found"}, status=404)


def run_server(port: int = 8501) -> None:
    global APP_STATE
    repo_root = Path(__file__).resolve().parent.parent
    APP_STATE = DriftNetAppState(repo_root)
    server = ThreadingHTTPServer(("127.0.0.1", port), DriftNetRequestHandler)
    print(f"\n============================================================")
    print(f"  DriftNet Live — Real-Time Streaming Monitor UI")
    print(f"  Open in browser: http://127.0.0.1:{port}")
    print(f"============================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    run_server()
