#!/usr/bin/env python3
"""Prepare relabeling: cross-group kNN consensus over CLAP embeddings splits
samples into (a) confidently-correct, (b) OBVIOUS mislabels (auto-fixed), and
(c) AMBIGUOUS (queued for human ear-review). Also picks a clean reference
example per class. Outputs to training_data/relabel/.
"""
import warnings; warnings.filterwarnings("ignore")
import os, csv, json, glob, numpy as np
TD="training_data"; OUT=f"{TD}/relabel"; os.makedirs(OUT,exist_ok=True)
K=15
d=np.load(f"{TD}/embeddings_v2.npz",allow_pickle=True)
X=np.array(d["X"],dtype=np.float32); y=np.array(d["y"]); paths=np.array(d["paths"])
groups=np.load(f"{TD}/groups_v2.npy")
# hash(12) -> disk path, for audio playback
disk={}
for p in glob.glob(f"{TD}/labeled/**/*",recursive=True)+glob.glob(f"{TD}/unsorted/**/*",recursive=True):
    if p.lower().endswith((".wav",".aif",".aiff",".flac",".mp3",".ogg")):
        disk.setdefault(os.path.splitext(os.path.basename(p))[0][:12],p)
def hid(p): return os.path.splitext(os.path.basename(str(p)))[0][:12]
orig={}
for r in csv.DictReader(open(f"{TD}/manifest.csv")): orig[r["hash"][:12]]=r["orig_name"]

Xn=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9); S=Xn@Xn.T; np.fill_diagonal(S,-1)
classes=sorted(set(y))
own=np.zeros(len(y)); cons=np.empty(len(y),dtype=object); cons_str=np.zeros(len(y)); dist=[None]*len(y)
for i in range(len(y)):
    nb=[j for j in np.argsort(-S[i]) if groups[j]!=groups[i]][:K]
    labs=y[nb]; own[i]=(labs==y[i]).mean()
    vals,cnts=np.unique(labs,return_counts=True); o=np.argsort(-cnts)
    cons[i]=vals[o[0]]; cons_str[i]=cnts[o[0]]/K
    dist[i]={str(vals[j]):int(cnts[j]) for j in o[:4]}

auto=[]; review=[]
for i in range(len(y)):
    if cons[i]==y[i]: continue                         # neighbors agree -> keep
    if own[i]<0.15 and cons_str[i]>=0.60:              # OBVIOUS mislabel
        auto.append({"hash":hid(paths[i]),"old":str(y[i]),"new":str(cons[i]),"own":round(float(own[i]),2)})
    elif own[i]<0.45 and hid(paths[i]) in disk:        # AMBIGUOUS -> human
        review.append({"hash":hid(paths[i]),"path":disk[hid(paths[i])],"current":str(y[i]),
                       "name":orig.get(hid(paths[i]),""),"neighbors":dist[i],"own":round(float(own[i]),2)})
# clean reference example per class = highest own-agreement sample that has audio
examples={}
for c in classes:
    ci=[i for i in np.where(y==c)[0] if hid(paths[i]) in disk]
    if ci:
        best=max(ci,key=lambda i: own[i]); examples[str(c)]={"hash":hid(paths[best]),"path":disk[hid(paths[best])]}
# most-uncertain first
review.sort(key=lambda r: r["own"])
json.dump(auto,open(f"{OUT}/auto_corrections.json","w"),indent=0)
json.dump(review,open(f"{OUT}/review_queue.json","w"),indent=0)
json.dump(examples,open(f"{OUT}/examples.json","w"))
# hash->path map for the server (only what the UI needs)
amap={r["hash"]:r["path"] for r in review}
for e in examples.values(): amap[e["hash"]]=e["path"]
json.dump(amap,open(f"{OUT}/audio_map.json","w"))
print(f"confidently correct: kept")
print(f"OBVIOUS mislabels (auto-fix): {len(auto)}")
print(f"AMBIGUOUS (human review):     {len(review)}")
print(f"reference examples picked for {len(examples)}/{len(classes)} classes")
