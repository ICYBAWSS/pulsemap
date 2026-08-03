#!/usr/bin/env python3
"""Export the shipped classifier: frozen CLAP -> StandardScaler -> contrastive
projection MLP (512->256->128) -> class-weighted logistic (128->20). Trained on
ALL cleaned+relabeled+augmented data (no held-out — this is the production
model). Writes native/models/model.json in the format classify.rs expects.
See RESULTS.md. Honest CV of this pipeline: ~75 balanced / ~85 overall.
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
TD="training_data"; RL=f"{TD}/relabel"; torch.manual_seed(0); np.random.seed(0)
d=np.load(f"{TD}/embeddings_v2.npz",allow_pickle=True)
X=np.array(d["X"],dtype=np.float32); y=np.array(d["y"]); paths=np.array(d["paths"])
def hid(p): return os.path.splitext(os.path.basename(str(p)))[0][:12]
h2i={hid(p):i for i,p in enumerate(paths)}
# apply auto-fixes + human review; drop 'notdrum'
drop=np.zeros(len(y),bool)
for a in json.load(open(f"{RL}/auto_corrections.json")):
    i=h2i.get(a["hash"]);
    if i is not None: y[i]=a["new"]
if os.path.exists(f"{RL}/decisions.jsonl"):
    for ln in open(f"{RL}/decisions.jsonl"):
        r=json.loads(ln); i=h2i.get(r["hash"])
        if i is None: continue
        if r["label"]=="notdrum": drop[i]=True
        elif r["label"] not in ("keep","skip"): y[i]=r["label"]
keep=~drop; X,y=X[keep],y[keep]
# augmentation (train-only data — fine, this is all-train)
aug=np.load(f"{TD}/aug_embeddings.npz",allow_pickle=True)
Xa,ya=np.array(aug["X"],dtype=np.float32),np.array(aug["y"])
Xall=np.vstack([X,Xa]); yall=np.concatenate([y,ya])
le=LabelEncoder().fit(yall); Y=le.transform(yall); NC=len(le.classes_)
print(f"train on {len(Xall)} ({len(X)} real + {len(Xa)} aug), {NC} classes")

sc=StandardScaler().fit(Xall); Xs=torch.tensor(sc.transform(Xall),dtype=torch.float32); yb=torch.tensor(Y)
cw=np.bincount(Y,minlength=NC).astype(np.float32); cwt=torch.tensor(cw.sum()/(NC*np.maximum(cw,1)))
net=nn.Sequential(nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.4),nn.Linear(256,128)); clf=nn.Linear(128,NC)
opt=torch.optim.AdamW(list(net.parameters())+list(clf.parameters()),lr=1e-3,weight_decay=1e-4)
for ep in range(60):
    net.train(); perm=torch.randperm(len(Xs))
    for s in range(0,len(Xs),256):
        bi=perm[s:s+256]; z=net(Xs[bi]); loss=F.cross_entropy(clf(z),yb[bi],weight=cwt)
        zn=F.normalize(z,dim=1); sim=zn@zn.T/0.1; lab=yb[bi]; pos=(lab[:,None]==lab[None,:]).float(); pos.fill_diagonal_(0)
        lsm=F.log_softmax(sim-1e9*torch.eye(len(bi)),dim=1); loss=loss+0.5*(-(pos*lsm).sum(1)/pos.sum(1).clamp(min=1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
net.eval()
with torch.no_grad(): Z=F.normalize(net(Xs),dim=1).numpy()
lr=LogisticRegression(C=1.0,max_iter=3000,class_weight="balanced").fit(Z,Y)

# Unsorted threshold: TRAIN confidence is over-optimistic (5th pct ~0.69), but on
# real/unseen sounds 0.50 gives ~5% to Unsorted (verified on test_samples) and
# matches the old well-tuned head. Field-tunable knob.
thr=0.50
print(f"MIN_CONFIDENCE (calibrated to ~5% on real data): {thr:.3f}")

w1,b1=net[0].weight.detach().numpy(), net[0].bias.detach().numpy()
w2,b2=net[3].weight.detach().numpy(), net[3].bias.detach().numpy()
out={"labels":list(le.classes_),
     "scaler_mean":sc.mean_.astype(float).tolist(),"scaler_scale":sc.scale_.astype(float).tolist(),
     "proj_w1":w1.astype(float).tolist(),"proj_b1":b1.astype(float).tolist(),
     "proj_w2":w2.astype(float).tolist(),"proj_b2":b2.astype(float).tolist(),
     "coef":lr.coef_.astype(float).tolist(),"intercept":lr.intercept_.astype(float).tolist(),
     "min_confidence":thr,"embedding_dim":512,"proj_dim":128,
     "head":"CLAP -> scaler -> MLP(512-256-128,relu) -> L2norm -> logistic(128-NC) -> softmax"}
os.makedirs(f"{TD}/../native/models",exist_ok=True)
json.dump(out,open("native/models/model.json","w"))
print(f"wrote native/models/model.json ({NC} labels): {list(le.classes_)}")
# sanity sample for Rust cross-check
np.save(f"{TD}/verify_input.npy",Xall[:3])
json.dump({"labels":list(le.classes_),"expect":[int(lr.predict(Z[i:i+1])[0]) for i in range(3)]},
          open(f"{TD}/verify_expect.json","w"))
print("DONE")
