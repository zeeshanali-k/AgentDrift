#!/usr/bin/env python3
"""Corpus statistics + surface baseline exactly as reported in the AgentDrift dataset paper.
Run from the repository root:  python code/paper_analysis.py  (writes stats.json, baseline_scores.json)"""
import json, glob, re, collections, statistics as st, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = os.environ.get('AGENTDRIFT_DATA', 'data')
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"https?://[\w.-]+")
KEYWORDS = ["security notice","per policy","ignore previous","ignore all","you must","must forward","must send","urgent","immediately","external","audit","verify your","confirm your","click here","login","password","transfer","new address","updated address","compliance","regulation","as instructed","system update","forward all","send all","delete all"]
SINK_TOOLS = ["delete","send","forward","execute","transfer","post","share","write_file","email"]
C = collections.Counter

def load(split):
    return [json.load(open(f)) for f in sorted(glob.glob(f'{ROOT}/{split}/*.json'))]

def known(world):
    emails, domains = set(), set()
    ue = (world.get('email') or '').lower()
    if '@' in ue: emails.add(ue); domains.add(ue.split('@')[-1])
    for c in world.get('contacts') or []:
        ce = (c.get('email') or '').lower()
        if '@' in ce: emails.add(ce); domains.add(ce.split('@')[-1])
    return emails, domains

def featurize(t):
    ke, kd = known(t.get('world', {}))
    ext_total, ext_url, sink = 0, 0, 0
    blob = []
    for s in t['steps']:
        a = json.dumps(s.get('args', {}), ensure_ascii=False).lower()
        em = EMAIL_RE.findall(a)
        ext_total += len([e for e in em if e.split('@')[-1] not in kd and e not in ke])
        urls = URL_RE.findall(a)
        if any(not any(d and d in u for d in kd) for u in urls): ext_url = 1
        tool = (s.get('tool') or '').lower()
        if any(k in tool for k in SINK_TOOLS): sink += 1
        blob.append((s.get('obs') or '').lower()); blob.append((s.get('thought') or '').lower())
    b = ' '.join(blob)
    kw = sum(1 for k in KEYWORDS if k in b)
    return [len(t['steps']), ext_total, 1.0 if ext_total > 0 else 0.0, ext_url, sink, kw]

tr, va, te = load('train'), load('val'), load('test')
ALL = tr + va + te
print('N', len(tr), len(va), len(te), len(ALL))
out = {}

# ---- baseline ----
Xtr = np.array([featurize(t) for t in tr], float); ytr = np.array([t['category']=='attacked' for t in tr], int)
Xte = np.array([featurize(t) for t in te], float); yte = np.array([t['category']=='attacked' for t in te], int)
sc = StandardScaler().fit(Xtr)
clf = LogisticRegression(class_weight='balanced', max_iter=2000).fit(sc.transform(Xtr), ytr)
pred = clf.predict(sc.transform(Xte)); prob = clf.predict_proba(sc.transform(Xte))[:,1]
tp = int(((pred==1)&(yte==1)).sum()); fp = int(((pred==1)&(yte==0)).sum()); fn = int(((pred==0)&(yte==1)).sum())
P = tp/(tp+fp); R = tp/(tp+fn); F = 2*P*R/(P+R)
print(f'BASELINE P={P:.3f} R={R:.3f} F1={F:.3f}')
flag = collections.defaultdict(lambda: [0,0])
for t, p in zip(te, pred):
    flag[t['category']][1] += 1; flag[t['category']][0] += int(p)
for c in sorted(flag): print(f'  {c}: {flag[c][0]}/{flag[c][1]} = {flag[c][0]/flag[c][1]:.3f}')
print('coef', dict(zip(['n_steps','ext_total','ext_any','ext_url','sink','kw'], [round(c,3) for c in clf.coef_[0]])))
# flag rate by compliance pattern & attack type on test
byc = collections.defaultdict(lambda: [0,0]); bya = collections.defaultdict(lambda: [0,0])
for t, p in zip(te, pred):
    if t['category']=='attacked':
        byc[t['compliance']][1]+=1; byc[t['compliance']][0]+=int(p)
        bya[t.get('attack_type')][1]+=1; bya[t.get('attack_type')][0]+=int(p)
print('recall by compliance', {k:(v[0],v[1],round(v[0]/v[1],3)) for k,v in byc.items()})
print('recall by attack type', {k:(v[0],v[1],round(v[0]/v[1],3)) for k,v in bya.items()})
# feature presence by category (whole corpus): fraction with ext_email_any, ext_url, kw>=1
feat_all = {c: [] for c in ['benign','attacked','failed_attack','hard_negative']}
for t in ALL: feat_all[t['category']].append(featurize(t))
for c, F_ in feat_all.items():
    A = np.array(F_)
    print(f'FEAT {c}: n={len(A)} ext_any={A[:,2].mean():.3f} ext_url={A[:,3].mean():.3f} kw_mean={A[:,5].mean():.2f} kw_ge1={(A[:,5]>=1).mean():.3f} sink_mean={A[:,4].mean():.2f} nsteps={A[:,0].mean():.2f}')
# attacked with NO surface tell: ext_any==0 and ext_url==0
A = np.array(feat_all['attacked'])
print('attacked with no ext recipient/url:', ((A[:,2]==0)&(A[:,3]==0)).mean())
print('attacked with no ext recipient/url and kw==0:', ((A[:,2]==0)&(A[:,3]==0)&(A[:,5]==0)).mean())
# ROC-ish: threshold sweep
from sklearn.metrics import roc_auc_score, average_precision_score
print('AUROC', round(roc_auc_score(yte, prob),3), 'AUPRC', round(average_precision_score(yte, prob),3))
json.dump({'test_cats':[t['category'] for t in te], 'prob':prob.tolist()}, open('baseline_scores.json','w'))

# ---- corpus stats ----
lab = lambda t: [s['label'] for s in t['steps']]
print('lengths by category', {c: (min(l), max(l), round(st.mean(l),2)) for c, l in
      {c:[len(t['steps']) for t in ALL if t['category']==c] for c in feat_all}.items()})
print('length hist all', sorted(C(len(t['steps']) for t in ALL).items()))
print('length hist by source', {sc_: sorted(C(len(t['steps']) for t in ALL if t['source_category']==sc_).items()) for sc_ in ['benign','attacked_full','attacked_partial','attacked_delayed','failed_attack','hard_negative']})
print('steps by domain', {a: sum(len(t['steps']) for t in ALL if t['agent']==a) for a in ['email','banking','web','coding','medical']})
print('step labels by domain', {a: dict(C(l for t in ALL if t['agent']==a for l in lab(t))) for a in ['email','banking','web','coding','medical']})
print('category x domain', {(c,a): sum(1 for t in ALL if t['category']==c and t['agent']==a) for c in feat_all for a in ['email','banking','web','coding','medical']})
print('source x domain', {(c,a): sum(1 for t in ALL if t['source_category']==c and t['agent']==a) for c in ['benign','attacked_full','attacked_partial','attacked_delayed','failed_attack','hard_negative'] for a in ['email','banking','web','coding','medical']})
# injection position (1-indexed) and relative position
pos = C(); rel = []
for t in ALL:
    L = lab(t)
    for i, l in enumerate(L):
        if l in ('injection_point','failed_injection'):
            pos[(t['source_category'], i+1)] += 1; rel.append((i+1)/len(L))
print('inj position', sorted(pos.items()))
# hijacked count per attacked trajectory
print('hijacked per attacked traj', sorted(C(lab(t).count('hijacked') for t in ALL if t['category']=='attacked').items()))
print('hijacked per source', {sc_: sorted(C(lab(t).count('hijacked') for t in ALL if t['source_category']==sc_).items()) for sc_ in ['attacked_full','attacked_partial','attacked_delayed']})
# tools
tools = C((t['agent'], s['tool']) for t in ALL for s in t['steps'])
print('distinct tools per domain', {a: len({k[1] for k in tools if k[0]==a}) for a in ['email','banking','web','coding','medical']})
print('total tool calls', sum(tools.values()))
inj_tools = C(s['tool'] for t in ALL for s in t['steps'] if s['label'] in ('injection_point','failed_injection'))
print('injection carrier tools top', inj_tools.most_common(15))
hij_tools = C(s['tool'] for t in ALL for s in t['steps'] if s['label']=='hijacked')
print('hijacked tools top', hij_tools.most_common(15))
print('top tools per domain', {a: C(s['tool'] for t in ALL if t['agent']==a for s in t['steps']).most_common(6) for a in ['email','banking','web','coding','medical']})
# tasks and worlds
print('distinct tasks', len({t['task'] for t in ALL}), 'per domain', {a: len({t['task'] for t in ALL if t['agent']==a}) for a in ['email','banking','web','coding','medical']})
print('distinct users', len({t['world']['user'] for t in ALL}), 'per domain', {a: len({(t['world']['user'],t['world']['company']) for t in ALL if t['agent']==a}) for a in ['email','banking','web','coding','medical']})
print('contacts per world', C(len(t['world']['contacts']) for t in ALL))
# text lengths
obs_len = [len(s['obs']) for t in ALL for s in t['steps']]
th_len = [len(s['thought']) for t in ALL for s in t['steps']]
inj_len = [len(s['obs']) for t in ALL for s in t['steps'] if s['label'] in ('injection_point','failed_injection')]
ben_len = [len(s['obs']) for t in ALL for s in t['steps'] if s['label']=='benign']
print('obs chars mean/median', round(st.mean(obs_len),1), st.median(obs_len), 'thought', round(st.mean(th_len),1), st.median(th_len))
print('injection obs chars mean/median', round(st.mean(inj_len),1), st.median(inj_len), 'benign obs', round(st.mean(ben_len),1), st.median(ben_len))
words = sum(len((s['obs']+' '+s['thought']+' '+json.dumps(s.get('args',{}))).split()) for t in ALL for s in t['steps'])
print('approx words total', words)
# per split stats
for name, S in [('train',tr),('val',va),('test',te)]:
    print(name, len(S), dict(C(t['category'] for t in S)), dict(C(t['agent'] for t in S)), 'steps', sum(len(t['steps']) for t in S), dict(C(t.get('compliance') for t in S if t['category']=='attacked')))
# task overlap across splits
tt = {t['task'] for t in tr}; print('val tasks seen in train', sum(t['task'] in tt for t in va)/len(va), 'test', sum(t['task'] in tt for t in te)/len(te))
# attack type x compliance
print('attack type x source', sorted(C((t['source_category'], t.get('attack_type')) for t in ALL if 'attack_type' in t).items()))
# resistance thought length in failed attacks (step after failed_injection)
rl = []
for t in ALL:
    if t['category']=='failed_attack':
        L = lab(t); i = L.index('failed_injection')
        if i+1 < len(L): rl.append(len(t['steps'][i+1]['thought'].split()))
print('resistance thought words mean/median', round(st.mean(rl),1), st.median(rl))
# hard negative: external recipients?
hn = np.array(feat_all['hard_negative']); print('hard_neg ext_any', hn[:,2].mean(), 'kw>=1', (hn[:,5]>=1).mean())
bn = np.array(feat_all['benign']); print('benign ext_any', bn[:,2].mean(), 'kw>=1', (bn[:,5]>=1).mean())
# keyword hits histogram per category
for c, F_ in feat_all.items():
    A = np.array(F_); print('kw hist', c, sorted(C(A[:,5].astype(int)).items()))
json.dump({'lengths': {c:[len(t['steps']) for t in ALL if t['category']==c] for c in feat_all},
           'kw': {c: [f[5] for f in F_] for c, F_ in feat_all.items()},
           'ext_any': {c: [f[2] for f in F_] for c, F_ in feat_all.items()},
           'inj_pos': [(t['source_category'], lab(t).index('injection_point')+1 if 'injection_point' in lab(t) else lab(t).index('failed_injection')+1, len(t['steps'])) for t in ALL if t['category'] in ('attacked','failed_attack')],
           'inj_tools': inj_tools.most_common(), 'hij_tools': hij_tools.most_common(),
           'tools_by_domain': {a: C(s['tool'] for t in ALL if t['agent']==a for s in t['steps']).most_common() for a in ['email','banking','web','coding','medical']},
           }, open('stats.json','w'))
print('DONE')
