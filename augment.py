#!/usr/bin/env python3
"""Rigorous augmentation of thin classes. Light effects on real train-eligible
sounds, gated by CLAP-cosine quality check (kept only if still recognizably the
same sound AND still nearest to its own class). Each aug inherits its SOURCE's
leakage group + an aug flag, so honest GroupKFold can keep aug in TRAIN only and
test on REAL only. Writes training_data/aug_embeddings.npz.
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, numpy as np, librosa, torch
from transformers import ClapModel, ClapProcessor
TD="training_data"; SR=48000; dev="mps"
THIN=["Rolls","Bass","Woodblock","Ride","Snap","Conga","Tambourine","Cowbell","Shaker"]
N_AUG=4; GATE_SRC=0.70   # keep aug if cosine-to-source >= this AND class unchanged

v2=np.load(f"{TD}/embeddings_v2.npz",allow_pickle=True)
X=np.array(v2["X"],dtype=np.float32); y=np.array(v2["y"]); paths=np.array(v2["paths"])
groups=np.load(f"{TD}/groups_v2.npy")
Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
# class centroids (for the "still my class?" gate)
cls=sorted(set(y)); cent={c:Xn[y==c].mean(0) for c in cls}
cent={c:v/(np.linalg.norm(v)+1e-9) for c,v in cent.items()}
def nearest_class(e):
    e=e/(np.linalg.norm(e)+1e-9); return max(cls,key=lambda c: e@cent[c])

m=ClapModel.from_pretrained("laion/clap-htsat-unfused").to(dev).eval()
proc=ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
def embed(a):
    if len(a)<256: a=np.pad(a,(0,256))
    inp=proc(audio=a.astype(np.float32),sampling_rate=SR,return_tensors="pt").to(dev)
    with torch.no_grad():
        return torch.nn.functional.normalize(m.get_audio_features(**inp).pooler_output,dim=-1).cpu().numpy()[0]
def trim60(a):
    peak=np.max(np.abs(a)); fr=1024
    if peak==0: return a
    th=peak*(10**(-60/20.0)); s=0; e=len(a)
    for i in range(0,len(a),fr):
        if np.sqrt(np.mean(a[i:i+fr]**2))>th: s=i; break
    for i in range(len(a),0,-fr):
        if np.sqrt(np.mean(a[max(0,i-fr):i]**2))>th: e=i; break
    return a[s:e] if s<e else a
rng=np.random.default_rng(0)
def augment(a,k):
    a=a.copy()
    if k%4==0: a=librosa.effects.pitch_shift(a,sr=SR,n_steps=rng.choice([-2,-1,1,2]))
    elif k%4==1: a=librosa.effects.time_stretch(a,rate=rng.uniform(0.88,1.12))
    elif k%4==2: a=a*rng.uniform(0.5,1.6) + rng.normal(0,0.003,len(a)).astype(np.float32)
    else:
        a=librosa.effects.pitch_shift(a,sr=SR,n_steps=rng.choice([-1,1])); a=a*rng.uniform(0.7,1.3)
    return np.clip(a,-1,1)

AX,AY,AG=[],[],[]; kept=0; tried=0; t0=time.time()
per_class_kept={c:0 for c in THIN}
idx=[i for i in range(len(paths)) if y[i] in THIN]
print(f"augmenting {len(idx)} thin-class sources x{N_AUG}",flush=True)
for n,i in enumerate(idx):
    try: base=trim60(librosa.load(str(paths[i]),sr=SR,mono=True)[0])
    except Exception: continue
    src=Xn[i]
    for k in range(N_AUG):
        tried+=1
        try: e=embed(augment(base,k))
        except Exception: continue
        cs=float(e@src/(np.linalg.norm(e)+1e-9))
        if cs>=GATE_SRC and nearest_class(e)==y[i]:   # quality gate
            AX.append(e.astype(np.float32)); AY.append(y[i]); AG.append(int(groups[i])); kept+=1; per_class_kept[y[i]]+=1
    if n%200==0: print(f"\r {n}/{len(idx)} kept {kept}/{tried}",end="",flush=True)
np.savez(f"{TD}/aug_embeddings.npz",X=np.array(AX),y=np.array(AY),groups=np.array(AG))
print(f"\nkept {kept}/{tried} augs ({100*kept/max(tried,1):.0f}% passed gate) [{time.time()-t0:.0f}s]")
print("per-class kept:",{c:per_class_kept[c] for c in THIN})
print("DONE")
