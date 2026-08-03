import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score
TD="training_data"; THIN=["Rolls","Bass","Woodblock","Ride","Snap","Conga","Tambourine","Cowbell","Shaker"]
v2=np.load(f"{TD}/embeddings_v2.npz",allow_pickle=True)
X=np.array(v2["X"],dtype=np.float32); ys=np.array(v2["y"]); groups=np.load(f"{TD}/groups_v2.npy")
aug=np.load(f"{TD}/aug_embeddings.npz",allow_pickle=True)
Xa=np.array(aug["X"],dtype=np.float32); yas=np.array(aug["y"]); ga=np.array(aug["groups"])
le=LabelEncoder().fit(ys); y=le.transform(ys); ya=le.transform(yas); NC=len(le.classes_)
def proj_cv(use_aug):
    yt,yp=[],[]
    for tr,te in GroupKFold(5).split(X,y,groups):
        trg=set(groups[tr]); Xtr,ytr=X[tr],y[tr]
        if use_aug:
            m=np.array([g in trg for g in ga]); Xtr=np.vstack([Xtr,Xa[m]]); ytr=np.concatenate([ytr,ya[m]])
        sc=StandardScaler().fit(Xtr); At=torch.tensor(sc.transform(Xtr),dtype=torch.float32); Bt=torch.tensor(sc.transform(X[te]),dtype=torch.float32); yb=torch.tensor(ytr)
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
        yt.append(y[te]); yp.append(lr.predict(Zte))
    yt,yp=np.concatenate(yt),np.concatenate(yp)
    thin=np.mean([100*(yp[yt==le.transform([c])[0]]==le.transform([c])[0]).mean() for c in THIN])
    return 100*(yt==yp).mean(),100*balanced_accuracy_score(yt,yp),thin
for tag,ua in [("contrastive (real)",False),("contrastive + aug",True)]:
    o,b,t=proj_cv(ua); print(f"{tag:<22} overall {o:.1f}%  balanced {b:.1f}%  thin {t:.1f}%",flush=True)
print("DONE")
