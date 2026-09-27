import json,numpy as np,pandas as pd,onnxruntime as ort
from PIL import Image
ann=pd.read_csv('/mnt/user-data/uploads/Annotations.csv',encoding='utf-8-sig')
ann['lesion']=ann.attributes.apply(json.loads).str.get('lesion').fillna('')
lab={}
for f,g in ann.groupby('filename'):
    s=set()
    for l in g.lesion:
        if l=='Lens dislocation/Cataract': s|={'Lens dislocation','Cataract'}
        elif l: s.add(l)
    lab[f]=s
meta=json.load(open('model/model_meta.json')); names=[t['name'] for t in meta['targets']]
sess=ort.InferenceSession('model/slid_effb0.onnx')
want=['Normal','Cataract','Intraocular lens','Lens dislocation','Keratitis','Corneal scarring','Corneal dystrophy','Corneal / Conjunctival tumor','Pinguecula','Pterygium','Subconjunctival hemorrhage','Pigmented nevus']
rng=np.random.default_rng(7); out=[]
for w in want:
    c=[f for f,s in lab.items() if (s==set() if w=='Normal' else (s=={w} or s=={w,'Conjunctival injection'}))]
    c=list(rng.choice(c,min(25,len(c)),replace=False)); sc=[]
    for f in c:
        im=Image.open(f'/home/claude/eye/cache/img512/{f[:-4]}.jpg').convert('RGB').resize((384,288),Image.BILINEAR)
        x=(np.asarray(im,np.float32)/255).transpose(2,0,1)[None]
        p=sess.run(['probs'],{'image':x})[0][0]
        k=0 if w=='Normal' else names.index(w)
        ok=(p[0]<meta['targets'][0]['threshold']) if w=='Normal' else p[k]>=meta['targets'][k]['threshold']
        sc.append((ok,p[k],f))
    good=[s for s in sc if s[0]]; good.sort(key=lambda s:s[1]); pick=good[len(good)//2]
    print(w,len(c),len(good),pick); out.append(dict(file=pick[2],label=w))
json.dump(out,open('picks.json','w'),indent=1)
