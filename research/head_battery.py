#!/usr/bin/env python3
"""Test whether a better HEAD (not encoder) squeezes more from frozen CLAP v2
embeddings, honest GroupKFold. Includes supervised-contrastive projection — the
'learn our own representation' idea at a scale that fits our data.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
TD="training_data"
d=np.load(f"{TD}/embeddings_v2.npz",allow_pickle=True)
X=np.array(d["X"],dtype=np.float32); ys=np.array(d["y"]); groups=np.load(f"{TD}/groups_v2.npy")
le=LabelEncoder(); y=le.fit_transform(ys); NC=len(le.classes_)
def report(name,yt,yp): print(f"{name:<34} overall {100*(yt==yp).mean():4.1f}%  balanced {100*balanced_accuracy_score(yt,yp):4.1f}%",flush=True)

def cv_sklearn(mk):
    yt,yp=[],[]
    for tr,te in GroupKFold(5).split(X,y,groups):
        c=mk(); c.fit(X[tr],y[tr]); yt.append(y[te]); yp.append(c.predict(X[te]))
    return np.concatenate(yt),np.concatenate(yp)

print(f"{len(X)} samples, {NC} classes. baseline is logistic.",flush=True)
yt,yp=cv_sklearn(lambda: make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000,class_weight="balanced"))); report("logistic (baseline)",yt,yp)
yt,yp=cv_sklearn(lambda: HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_depth=6,class_weight="balanced")); report("HistGradientBoosting (tree)",yt,yp)

# torch MLP head
dev="cpu"
def cv_torch(project_then_logistic):
    yt,yp=[],[]
    for tr,te in GroupKFold(5).split(X,y,groups):
        sc=StandardScaler().fit(X[tr]); Xtr=torch.tensor(sc.transform(X[tr]),dtype=torch.float32); Xte=torch.tensor(sc.transform(X[te]),dtype=torch.float32)
        ytr=torch.tensor(y[tr])
        cw=np.bincount(y[tr],minlength=NC).astype(np.float32); cw=torch.tensor(cw.sum()/(NC*np.maximum(cw,1)))
        net=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.4),nn.Linear(256,128))
        clf=nn.Linear(128,NC)
        opt=torch.optim.AdamW(list(net.parameters())+list(clf.parameters()),lr=1e-3,weight_decay=1e-4)
        for ep in range(60):
            net.train(); perm=torch.randperm(len(Xtr))
            for s in range(0,len(Xtr),256):
                b=perm[s:s+256]; z=net(Xtr[b]); logit=clf(z)
                loss=F.cross_entropy(logit,ytr[b],weight=cw)
                if project_then_logistic:  # supervised contrastive term on the projection
                    zn=F.normalize(z,dim=1); sim=zn@zn.T/0.1
                    lab=ytr[b]; pos=(lab[:,None]==lab[None,:]).float(); pos.fill_diagonal_(0)
                    lsm=F.log_softmax(sim-1e9*torch.eye(len(b)),dim=1)
                    scl=-(pos*lsm).sum(1)/pos.sum(1).clamp(min=1)
                    loss=loss+0.5*scl.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            if project_then_logistic:
                Ztr=F.normalize(net(Xtr),dim=1).numpy(); Zte=F.normalize(net(Xte),dim=1).numpy()
                lr=LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced").fit(Ztr,y[tr])
                pred=lr.predict(Zte)
            else:
                pred=clf(net(Xte)).argmax(1).numpy()
        yt.append(y[te]); yp.append(pred)
    return np.concatenate(yt),np.concatenate(yp)

yt,yp=cv_torch(False); report("MLP head",yt,yp)
yt,yp=cv_torch(True); report("supervised-contrastive projection",yt,yp)
print("DONE",flush=True)
