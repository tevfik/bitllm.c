import json, sys, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
d=sys.argv[1]
tok=AutoTokenizer.from_pretrained(d); m=AutoModelForSequenceClassification.from_pretrained(d, torch_dtype=torch.float32).eval()
pairs=json.load(open(sys.argv[2]))
out=[]
with torch.no_grad():
    for q,doc in pairs:
        enc=tok(q,doc,return_tensors='pt',truncation=True,max_length=8192)
        out.append({"ids":enc['input_ids'][0].tolist(),"logit":float(m(**enc).logits.view(-1)[0])})
json.dump(out,open(sys.argv[3],'w'))
