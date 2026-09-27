import json, numpy as np, pandas as pd, torch, timm, torch.nn as nn
from sklearn.mixture import GaussianMixture
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
C='/home/claude/eye/cache'
ann=pd.read_csv('/mnt/user-data/uploads/Annotations.csv',encoding='utf-8-sig')
ann['lesion']=ann.attributes.apply(json.loads).str.get('lesion').fillna('')
LES=["Cataract","Intraocular lens","Lens dislocation","Keratitis","Corneal scarring","Corneal dystrophy","Corneal / Conjunctival tumor","Pinguecula","Pterygium","Subconjunctival hemorrhage","Conjunctival injection","Conjunctival cyst","Pigmented nevus"]
files=sorted(ann.filename.unique(),key=lambda f:int(f[:-4]))
lab={}
for f,g in ann.groupby('filename'):
    s=set()
    for l in g.lesion:
        if l=='Lens dislocation/Cataract': s|={'Lens dislocation','Cataract'}
        elif l: s.add(l)
    lab[f]=frozenset(s)
Y=np.array([[int(l in lab[f]) for l in LES] for f in files]); A=(Y.sum(1)>0).astype(int)
def prim(s):
    if not s: return 'Normal'
    if s==frozenset({'Keratitis','Conjunctival injection'}): return 'Keratitis'
    return next(iter(s)) if len(s)==1 else 'Multi'
P=np.array([prim(lab[f]) for f in files])
E=np.load(f'{C}/emb_b0.npy'); En=E/np.linalg.norm(E,axis=1,keepdims=True)
ls=np.array(['|'.join(sorted(lab[f])) for f in files])
adj=(En[:-1]*En[1:]).sum(1); same=ls[:-1]==ls[1:]
g=GaussianMixture(2,random_state=42).fit(adj[same,None]); grid=np.linspace(0,1,2001); hi=int(np.argmax(g.means_.ravel()))
tau=float(grid[np.argmax(g.predict_proba(grid[:,None])[:,hi]>0.5)])
m=(adj>=tau)&same; I=np.where(m)[0]
G=connected_components(coo_matrix((np.ones(len(I)),(I,I+1)),shape=(len(files),)*2),directed=False)[1]
print('tau',tau,'groups',G.max()+1)
T=np.column_stack([A,Y]); names=['Abnormal']+LES
folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(E,P,G))
mk=lambda: LogisticRegression(C=0.05,max_iter=3000,class_weight='balanced')
meta=[]; W=[]; B=[]
for k,n in enumerate(names):
    y=T[:,k]; oof=np.zeros(len(y))
    for tr,te in folds:
        sc=StandardScaler().fit(E[tr]); oof[te]=mk().fit(sc.transform(E[tr]),y[tr]).predict_proba(sc.transform(E[te]))[:,1]
    fpr,tpr,thr=roc_curve(y,oof); j=np.where(1-fpr>=0.90)[0][-1]
    sc=StandardScaler().fit(E); lr=mk().fit(sc.transform(E),y)
    w=lr.coef_.ravel()/sc.scale_; b=lr.intercept_[0]-(sc.mean_*w).sum()
    W.append(w); B.append(b)
    meta.append(dict(name=n,auroc=round(roc_auc_score(y,oof),3),threshold=float(thr[j]),sens_at_spec90=round(float(tpr[j]),3),prevalence=round(float(y.mean()),3),n_pos=int(y.sum())))
    print(meta[-1])
W=np.array(W,np.float32); B=np.array(B,np.float32)
# OOD: centroids of normalized embeddings per primary class
cls=sorted(set(P)); cent=np.array([En[P==c].mean(0) for c in cls]); cent/=np.linalg.norm(cent,axis=1,keepdims=True)
sim=(En@cent.T).max(1); ood_thr=float(np.percentile(sim,1))
print('ood thr',ood_thr, np.percentile(sim,[1,5,50]))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.bb=timm.create_model('efficientnet_b0',pretrained=False,num_classes=0)
        s.bb.load_state_dict(torch.load(f'{C}/efficientnet_b0_ra-3dd342df.pth'),strict=False)
        s.register_buffer('W',torch.tensor(W)); s.register_buffer('b',torch.tensor(B))
        s.register_buffer('mean',torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)); s.register_buffer('std',torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))
    def forward(s,x):
        fm=s.bb.forward_features((x-s.mean)/s.std); pooled=fm.mean((2,3))
        logits=pooled@s.W.T+s.b; cams=torch.einsum('kc,bchw->bkhw',s.W,fm)
        return torch.sigmoid(logits),cams,pooled
net=Net().eval()
x=torch.rand(1,3,288,384)
torch.onnx.export(net,x,'model/slid_effb0.onnx',input_names=['image'],output_names=['probs','cams','embedding'],opset_version=17,dynamo=False)
# check parity with numpy on a real image
from PIL import Image
im=np.asarray(Image.open(f'{C}/img512/1000.jpg').convert('RGB').resize((384,288),Image.BILINEAR),np.float32)/255
xt=torch.tensor(im.transpose(2,0,1))[None]
with torch.no_grad(): p,c,e=net(xt)
import onnxruntime as ort
s=ort.InferenceSession('model/slid_effb0.onnx'); po,co,eo=s.run(None,{'image':xt.numpy()})
print('parity',np.abs(po-p.numpy()).max(), co.shape)
json.dump(dict(labels=names,input_hw=[288,384],targets=meta,ood=dict(classes=cls,centroids=cent.round(5).tolist(),threshold=ood_thr),tau=tau,n_groups=int(G.max()+1)),open('model/model_meta.json','w'))
