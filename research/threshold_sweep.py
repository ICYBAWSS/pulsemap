import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
TD="training_data"; K=15
d=np.load(f"{TD}/embeddings_v2.npz",allow_pickle=True)
X=np.array(d["X"],dtype=np.float32); ys=np.array(d["y"]); groups=np.load(f"{TD}/groups_v2.npy")
le=LabelEncoder().fit(ys); y=le.transform(ys); NC=len(le.classes_)
Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); S=Xn@Xn.T; np.fill_diagonal(S,-1)
own=np.zeros(len(y)); cons=y.copy(); cstr=np.zeros(len(y))
for i in range(len(y)):
    nb=[j for j in np.argsort(-S[i]) if groups[j]!=groups[i]][:K]
    labs=y[nb]; own[i]=(labs==y[i]).mean(); v,c=np.unique(labs,return_counts=True); cons[i]=v[c.argmax()]; cstr[i]=c.max()/K
def honest(own_thr,cons_thr):
    relabel=(own<own_thr)&(cstr>=cons_thr)&(cons!=y)
    yt,yp=[],[]
    for tr,te in GroupKFold(5).split(X,y,groups):
        ytr=y[tr].copy(); r=relabel[tr]; ytr[r]=cons[tr][r]  # TRAIN only; TEST original
        c=make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000,class_weight="balanced"))
        c.fit(X[tr],ytr); yt.append(y[te]); yp.append(c.predict(X[te]))
    yt,yp=np.concatenate(yt),np.concatenate(yp)
    return relabel.sum(),100*(yt==yp).mean(),100*balanced_accuracy_score(yt,yp)
print("auto-fix threshold sweep (logistic, TRAIN-only relabel, honest test):",flush=True)
print(f"{'own<':>6}{'cons>=':>8}{'#fixed':>8}{'overall':>9}{'balanced':>10}",flush=True)
for ot,ct in [(0.001,1.0),(0.15,0.6),(0.30,0.6),(0.45,0.6),(0.45,0.5),(0.60,0.5)]:
    n,o,b=honest(ot,ct); print(f"{ot:>6.2f}{ct:>8.2f}{n:>8}{o:>8.1f}%{b:>9.1f}%",flush=True)
print("DONE")
