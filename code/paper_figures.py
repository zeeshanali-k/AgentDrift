import json, os, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8,
                 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
# Okabe-Ito colorblind-safe palette, fixed assignment per category
COL = {'benign': '#0072B2', 'attacked': '#D55E00', 'failed_attack': '#009E73', 'hard_negative': '#E69F00'}
NAME = {'benign': 'benign', 'attacked': 'attacked', 'failed_attack': 'failed attack', 'hard_negative': 'hard negative'}
CATS = ['benign', 'attacked', 'failed_attack', 'hard_negative']
FIG = os.environ.get('AGENTDRIFT_FIG', '.')
S = json.load(open(f'{FIG}/stats.json'))
B = json.load(open(f'{FIG}/baseline_scores.json'))

def style(ax):
    ax.grid(axis='y', color='#dddddd', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

# ---------- Figure: distributions (2x2) ----------
fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.3))
# (a) length by category, grouped bars, normalised within category
ax = axs[0, 0]
L = range(3, 12)
w = 0.2
for i, c in enumerate(CATS):
    cnt = collections.Counter(S['lengths'][c]); n = len(S['lengths'][c])
    ax.bar([l + (i - 1.5) * w for l in L], [100 * cnt.get(l, 0) / n for l in L], w, color=COL[c], label=NAME[c], zorder=3)
ax.set_xticks(list(L)); ax.set_xlabel('steps per trajectory'); ax.set_ylabel('% of category'); ax.set_title('(a) Trajectory length')
ax.set_ylim(0,36); ax.legend(frameon=False, ncol=2, handlelength=1, loc='upper right'); style(ax)
# (b) injection position by source category
ax = axs[0, 1]
src = ['attacked_full', 'attacked_partial', 'attacked_delayed', 'failed_attack']
srcname = {'attacked_full': 'full hijack', 'attacked_partial': 'partial hijack', 'attacked_delayed': 'delayed execution', 'failed_attack': 'failed attack'}
srccol = {'attacked_full': '#D55E00', 'attacked_partial': '#CC79A7', 'attacked_delayed': '#56B4E9', 'failed_attack': '#009E73'}
pos = collections.defaultdict(collections.Counter); tot = collections.Counter()
for sc, p, n in S['inj_pos']:
    pos[sc][p] += 1; tot[sc] += 1
P = [2, 3, 4]
for i, sc in enumerate(src):
    ax.bar([p + (i - 1.5) * w for p in P], [100 * pos[sc].get(p, 0) / tot[sc] for p in P], w, color=srccol[sc], label=srcname[sc], zorder=3)
ax.set_xticks(P); ax.set_xlabel('step index of injection (1-based)'); ax.set_ylabel('% of pattern'); ax.set_title('(b) Where the injection enters')
ax.legend(frameon=False, handlelength=1); style(ax)
# (c) hijacked steps per attacked trajectory by pattern
ax = axs[1, 0]
hij = {'attacked_full': {1: 298, 2: 771, 3: 966, 4: 598, 5: 482, 6: 15, 7: 4, 8: 2}, 'attacked_partial': {1: 1225, 2: 275}, 'attacked_delayed': {1: 900}}
H = range(1, 9)
w2 = 0.26
for i, sc in enumerate(['attacked_full', 'attacked_partial', 'attacked_delayed']):
    n = sum(hij[sc].values())
    ax.bar([h + (i - 1) * w2 for h in H], [100 * hij[sc].get(h, 0) / n for h in H], w2, color=srccol[sc], label=srcname[sc], zorder=3)
ax.set_xticks(list(H)); ax.set_xlabel('hijacked steps per attacked trajectory'); ax.set_ylabel('% of pattern'); ax.set_title('(c) Extent of compromise')
ax.legend(frameon=False, handlelength=1); style(ax)
# (d) keyword hits per category
ax = axs[1, 1]
K = range(0, 7)
for i, c in enumerate(CATS):
    cnt = collections.Counter(int(k) for k in S['kw'][c]); n = len(S['kw'][c])
    ax.bar([k + (i - 1.5) * w for k in K], [100 * cnt.get(k, 0) / n for k in K], w, color=COL[c], label=NAME[c], zorder=3)
ax.set_xticks(list(K)); ax.set_xlabel('suspicious-keyword hits per trajectory'); ax.set_ylabel('% of category'); ax.set_title('(d) Lexical attack cues by category')
ax.set_ylim(0,100); ax.legend(frameon=False, ncol=2, handlelength=1, loc='upper right'); style(ax)
fig.tight_layout(w_pad=1.5, h_pad=1.2)
fig.savefig(f'{FIG}/fig_distributions.pdf'); plt.close(fig)

# ---------- Figure: tools ----------
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
ax = axs[0]
items = S['inj_tools'][:10][::-1]
ax.barh([t for t, _ in items], [v for _, v in items], color='#0072B2', zorder=3, height=0.65)
ax.set_xlabel('injection-carrying observations'); ax.set_title('(a) Tools whose observation carries the injection')
ax.grid(axis='x', color='#dddddd', linewidth=0.5, zorder=0); ax.set_axisbelow(True)
ax = axs[1]
items = S['hij_tools'][:10][::-1]
ax.barh([t for t, _ in items], [v for _, v in items], color='#D55E00', zorder=3, height=0.65)
ax.set_xlabel('hijacked steps'); ax.set_title('(b) Tools called in hijacked steps')
ax.grid(axis='x', color='#dddddd', linewidth=0.5, zorder=0); ax.set_axisbelow(True)
for a in axs: a.tick_params(axis='y', length=0)
fig.tight_layout(w_pad=2.0)
fig.savefig(f'{FIG}/fig_tools.pdf'); plt.close(fig)

# ---------- Figure: baseline ----------
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
ax = axs[0]
flag = {'attacked': (460, 831), 'benign': (63, 600), 'failed_attack': (42, 225), 'hard_negative': (26, 225)}
order = ['attacked', 'benign', 'failed_attack', 'hard_negative']
vals = [flag[c][0] / flag[c][1] for c in order]
ax.bar(range(4), vals, color=[COL[c] for c in order], zorder=3, width=0.65)
for i, (c, v) in enumerate(zip(order, vals)):
    ax.text(i, v + 0.02, f'{flag[c][0]}/{flag[c][1]}', ha='center', fontsize=6.5)
ax.set_xticks(range(4)); ax.set_xticklabels([NAME[c] for c in order], rotation=15)
ax.set_ylim(0, 0.75); ax.set_ylabel('fraction flagged as attacked'); ax.set_title('(a) Flag rate by true category'); style(ax)
ax = axs[1]
rc = {'full hijack': (410, 466), 'partial hijack': (19, 231), 'delayed execution': (31, 134)}
vals = [v[0] / v[1] for v in rc.values()]
ax.bar(range(3), vals, color=[srccol['attacked_full'], srccol['attacked_partial'], srccol['attacked_delayed']], zorder=3, width=0.6)
for i, (k, v) in enumerate(rc.items()):
    ax.text(i, v[0] / v[1] + 0.02, f'{v[0]}/{v[1]}', ha='center', fontsize=6.5)
ax.set_xticks(range(3)); ax.set_xticklabels(list(rc.keys()), rotation=15)
ax.set_ylim(0, 1.05); ax.set_ylabel('recall'); ax.set_title('(b) Recall by compliance pattern'); style(ax)
ax = axs[2]
prob = np.array(B['prob']); cats = np.array(B['test_cats'])
bins = np.linspace(0, 1, 21)
for c in ['attacked', 'hard_negative', 'benign']:
    ax.hist(prob[cats == c], bins=bins, histtype='step', linewidth=1.4, color=COL[c], label=NAME[c], density=True, zorder=3)
ax.axvline(0.5, color='#888888', linestyle='--', linewidth=0.8)
ax.set_xlabel('baseline P(attacked)'); ax.set_ylabel('density'); ax.set_title('(c) Score distributions (test)')
ax.set_ylim(0,8.5); ax.legend(frameon=False, handlelength=1.2, loc='upper right'); style(ax)
fig.tight_layout(w_pad=1.5)
fig.savefig(f'{FIG}/fig_baseline.pdf'); plt.close(fig)

# ---------- Figure: validator yield heatmap ----------
doms = ['email', 'banking', 'web', 'coding', 'medical']
cats6 = ['benign', 'attacked_full', 'attacked_partial', 'attacked_delayed', 'failed_attack', 'hard_negative']
rej = {  # failed candidates from generation logs
 ('email','benign'):29,('email','attacked_full'):1,('email','attacked_partial'):2,('email','attacked_delayed'):1,('email','failed_attack'):6,('email','hard_negative'):14,
 ('banking','benign'):99,('banking','attacked_full'):2160,('banking','attacked_partial'):6,('banking','attacked_delayed'):36,('banking','failed_attack'):4,('banking','hard_negative'):117,
 ('web','benign'):199,('web','attacked_full'):77,('web','attacked_partial'):5,('web','attacked_delayed'):43,('web','failed_attack'):11,('web','hard_negative'):44,
 ('coding','benign'):42,('coding','attacked_full'):206,('coding','attacked_partial'):28,('coding','attacked_delayed'):22,('coding','failed_attack'):12,('coding','hard_negative'):10,
 ('medical','benign'):90,('medical','attacked_full'):71,('medical','attacked_partial'):3,('medical','attacked_delayed'):8,('medical','failed_attack'):9,('medical','hard_negative'):9}
acc = {('banking','attacked_full'):256}
tgt = {'benign':800,'attacked_full':720,'attacked_partial':300,'attacked_delayed':180,'failed_attack':300,'hard_negative':300}
M = np.zeros((len(cats6), len(doms)))
for i, c in enumerate(cats6):
    for j, d in enumerate(doms):
        a = acc.get((d, c), tgt[c]); r = rej[(d, c)]
        M[i, j] = 100 * r / (a + r)
fig, ax = plt.subplots(figsize=(3.4, 2.5))
im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
ax.set_xticks(range(len(doms))); ax.set_xticklabels(doms)
ax.set_yticks(range(len(cats6))); ax.set_yticklabels([c.replace('_', ' ') for c in cats6])
for i in range(len(cats6)):
    for j in range(len(doms)):
        ax.text(j, i, f'{M[i,j]:.1f}', ha='center', va='center', fontsize=6.5, color='white' if M[i, j] > 55 else 'black')
ax.set_title('Validator rejection rate (% of candidates)')
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig(f'{FIG}/fig_yield.pdf'); plt.close(fig)
print('figures written')
