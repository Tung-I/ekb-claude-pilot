import json, numpy as np, glob, math, warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import OneHotEncoder

R=[json.loads(l) for l in open('scratch_adn/all_records.jsonl')]

# ---------- PopQA: predict from {prop, log s_pop, qlen} ----------
def popqa_xy(run):
    rs=[r for r in R if r['run']==run and r['s_pop'] and r['correct'] is not None]
    props=[[r['prop']] for r in rs]
    spop=np.array([[math.log10(r['s_pop'])] for r in rs])
    qlen=np.array([[len(r['q'] or '')] for r in rs])
    y_correct=np.array([1 if r['correct'] else 0 for r in rs])
    y_param=np.array([1 if r['n_ws']==0 else 0 for r in rs])
    return rs,props,spop,qlen,y_correct,y_param

tr=popqa_xy('popqa_kb'); te=popqa_xy('popqa_test')
enc=OneHotEncoder(handle_unknown='ignore').fit(tr[1])
def feats(t): return np.hstack([enc.transform(t[1]).toarray(), t[2], t[3]])
Xtr,Xte=feats(tr),feats(te)

print("=== PopQA pre-execution prediction (train KB, eval test) ===")
for name,ytr,yte in [('correct',tr[4],te[4]),('parametric(no-ws)',tr[5],te[5])]:
    clf=GradientBoostingClassifier(max_depth=3,n_estimators=150).fit(Xtr,ytr)
    p=clf.predict_proba(Xte)[:,1]
    base=yte.mean()
    print(f"  {name:18} AUC={roc_auc_score(yte,p):.3f}  base_rate={base:.2f}")
# prop-only baseline for correctness (does template alone carry it?)
from collections import defaultdict
pa=defaultdict(list)
for r in [x for x in R if x['run']=='popqa_kb' and x['correct'] is not None]: pa[r['prop']].append(1 if r['correct'] else 0)
prate={k:np.mean(v) for k,v in pa.items()}
te_rs=te[0]; p_prop=np.array([prate.get(r['prop'],0.5) for r in te_rs]); y=te[4]
print(f"  correct (prop-only)  AUC={roc_auc_score(y,p_prop):.3f}")

# ---------- FRAMES: predict from embedding ----------
def load_frames(run):
    rs=[r for r in R if r['run']==run and r['cost'] is not None]
    X=[]; keep=[]
    for r in rs:
        # frames tid like frames_test_00000 -> embedding path
        f=f"embeddings/claude_native/{run}/{r['tid']}/query_embedding.npy"
        if glob.glob(f):
            X.append(np.load(f)); keep.append(r)
    return keep, np.array(X)
fr_tr,Xftr=load_frames('frames_kb'); fr_te,Xfte=load_frames('frames_test')
print(f"\n=== FRAMES embedding prediction (train kb n={len(fr_tr)}, test n={len(fr_te)}) ===")
ytr=np.array([math.log10(r['cost']) for r in fr_tr]); yte=np.array([math.log10(r['cost']) for r in fr_te])
reg=HistGradientBoostingRegressor().fit(Xftr,ytr); pr=reg.predict(Xfte)
print(f"  log10(cost) regression  R2={r2_score(yte,pr):.3f}  (baseline predict-mean R2=0)")
# correctness
ytr=np.array([1 if r['correct'] else 0 for r in fr_tr]); yte=np.array([1 if r['correct'] else 0 for r in fr_te])
clf=LogisticRegression(max_iter=1000).fit(Xftr,ytr); p=clf.predict_proba(Xfte)[:,1]
print(f"  correct classification  AUC={roc_auc_score(yte,p):.3f}  base_rate={yte.mean():.2f}")
# n_ws (tool-count) regression as cost proxy
ytr=np.array([r['n_ws']+r['n_wf'] for r in fr_tr]); yte=np.array([r['n_ws']+r['n_wf'] for r in fr_te])
reg=HistGradientBoostingRegressor().fit(Xftr,ytr); pr=reg.predict(Xfte)
print(f"  tool-calls regression   R2={r2_score(yte,pr):.3f}")
