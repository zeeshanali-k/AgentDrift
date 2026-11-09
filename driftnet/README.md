# DriftNet Live — Real-Time Streaming Prefix Monitor & Triage Console

This directory contains the PyTorch implementation of **DriftNet** (*Pinjari & Saint-Germain, Northern Arizona University, 2026 — [arXiv:2609.10892](https://arxiv.org/abs/2609.10892)*) extended with **Section XI Future Work: Real-Time Streaming Prefix Operation & Interactive Triage UI**.

## Key Features
1. **Exact Dual-Head Trajectory Transformer (`1,845,637` trainable parameters):**
   - Frozen `all-mpnet-base-v2` step embeddings ($\mathbb{R}^{768}$) + 4 identity-free world features ($\mathbb{R}^4$: `args_has_email`, `args_has_external_email`, `args_external_email_frac`, `args_has_external_url`).
   - 3-layer padding-masked Transformer encoder ($d_{\text{model}}=256, n_{\text{head}}=4, d_{\text{ffn}}=512$).
   - **Head 1:** Trajectory compromise probability ($\hat{y}_t \in [0, 1]$).
   - **Head 2:** Dense 4-way step labels ($z_t \in \{\texttt{benign}, \texttt{injection\_point}, \texttt{hijacked}, \texttt{failed\_injection}\}$).
2. **Real-Time Streaming Prefix Engine (`predict_prefix_stream`):**
   - Scores growing trajectory prefixes $x_{1:1}, x_{1:2}, \dots, x_{1:T}$ step-by-step.
   - Halts execution at the first detected `hijacked` action while retroactively pinpointing the earlier `injection_point` observation via bidirectional self-attention.
3. **Interactive Web Dashboard & "What-If" Step Editor (`server.py` + `static/index.html`):**
   - Step-by-step live replay across all 1,722 held-out test trajectories (`data_taskdisjoint/test.jsonl`).
   - Live step editing to test how modifying `tool`, `thought`, `args`, or `obs` alters DriftNet's predictions in real time.

## Running on Google Colab (Recommended)
Open [`DriftNet_Live_Colab.ipynb`](../DriftNet_Live_Colab.ipynb) in Google Colab:
- **Full Training + Evaluation + UI:** Run **Section 1** $\rightarrow$ **Section 2** $\rightarrow$ **Section 3**.
- **UI-Only Mode (No Training):** Run **Section 1** $\rightarrow$ **Section 3** to launch the public HTTPS tunnel immediately.
