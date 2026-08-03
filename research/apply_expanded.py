import warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
relabel=(own<0.45)&(cstr>=0.6)&(cons!=y); print(f"expanded auto-fix: {relabel.sum()} samples",flush=True)
def stack(train_relabel):
    yt,yp=[],[]
    for tr,te in GroupKFold(5).split(X,y,groups):
        ytr=y[tr].copy()
        if train_relabel: r=relabel[tr]; ytr[r]=cons[tr][r]
        sc=StandardScaler().fit(X[tr]); At=torch.tensor(sc.transform(X[tr]),dtype=torch.float32); Bt=torch.tensor(sc.transform(X[te]),dtype=torch.float32); yb=torch.tensor(ytr)
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
    return 100*(yt==yp).mean(),100*balanced_accuracy_score(yt,yp)
o,b=stack(True); print(f"expanded auto-fix + contrastive (HONEST, train-only): overall {o:.1f}%  balanced {b:.1f}%")
print("DONE")
