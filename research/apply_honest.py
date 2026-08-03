import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score
TD="training_data"; RL=f"{TD}/relabel"
d=np.load(f"{TD}/embeddings_v2.npz",allow_pickle=True)
X=np.array(d["X"],dtype=np.float32); y=np.array(d["y"]); paths=np.array(d["paths"]); groups=np.load(f"{TD}/groups_v2.npy")
def hid(p): return os.path.splitext(os.path.basename(str(p)))[0][:12]
h2i={hid(p):i for i,p in enumerate(paths)}
# corrected label per sample (auto + human); 'notdrum' -> drop
new=y.copy(); drop=np.zeros(len(y),bool); is_human=np.zeros(len(y),bool)
for a in json.load(open(f"{RL}/auto_corrections.json")):
    i=h2i.get(a["hash"]);
    if i is not None: new[i]=a["new"]
if os.path.exists(f"{RL}/decisions.jsonl"):
    for ln in open(f"{RL}/decisions.jsonl"):
        r=json.loads(ln); i=h2i.get(r["hash"])
        if i is None: continue
        if r["label"]=="notdrum": drop[i]=True
        elif r["label"] in ("keep","skip"): pass
        else: new[i]=r["label"]; is_human[i]=True
le=LabelEncoder().fit(np.concatenate([y,new])); Yo=le.transform(y); Yn=le.transform(new); NC=len(le.classes_)
keep=~drop
def stack(train_labels, test_labels):
    yt,yp=[],[]
    for tr,te in GroupKFold(5).split(X,Yo,groups):
        tr=tr[keep[tr]]; te=te[keep[te]]
        sc=StandardScaler().fit(X[tr]); At=torch.tensor(sc.transform(X[tr]),dtype=torch.float32); Bt=torch.tensor(sc.transform(X[te]),dtype=torch.float32)
        ytr=train_labels[tr]; yb=torch.tensor(ytr)
        cw=np.bincount(ytr,minlength=NC).astype(np.float32); cw=torch.tensor(cw.sum()/(NC*np.maximum(cw,1)))
        net=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.4),nn.Linear(256,128)); clf=nn.Linear(128,NC)
        opt=torch.optim.AdamW(list(net.parameters())+list(clf.parameters()),lr=1e-3,weight_decay=1e-4)
        for ep in range(60):
            net.train(); perm=torch.randperm(len(At))
            for s in range(0,len(At),256):
                bi=perm[s:s+256]; z=net(At[bi]); loss=F.cross_entropy(clf(z),yb[bi],weight=cw)
                zn=F.normalize(z,dim=1); sim=zn@zn.T/0.1; lab=yb[bi]; pos=(lab[:,None]==lab[None,:]).float(); pos.fill_diagonal_(0)
                lsm=F.log_softmax(sim-1e9*torch.eye(len(bi)),dim=1); loss=loss+0.5*(-(pos*lsm).sum(1)/pos.sum(1).clamp(min=1)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): Ztr=F.normalize(net(At),dim=1).numpy(); Zte=F.normalize(net(Bt),dim=1).numpy()
        lr=LogisticRegression(C=1.0,max_iter=2000,class_weight="balanced").fit(Ztr,ytr)
        yt.append(test_labels[te]); yp.append(lr.predict(Zte))
    yt,yp=np.concatenate(yt),np.concatenate(yp)
    return 100*(yt==yp).mean(),100*balanced_accuracy_score(yt,yp)
o,b=stack(Yo,Yo); print(f"1. baseline (no relabel):                overall {o:.1f}%  balanced {b:.1f}%",flush=True)
o,b=stack(Yn,Yo); print(f"2. HONEST (relabel TRAIN only, test orig): overall {o:.1f}%  balanced {b:.1f}%",flush=True)
o,b=stack(Yn,Yn); print(f"3. optimistic (relabel all - circular):   overall {o:.1f}%  balanced {b:.1f}%",flush=True)
print("DONE")
