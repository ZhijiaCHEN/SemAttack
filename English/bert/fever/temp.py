import json
from tqdm.auto import tqdm
label_dict = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}

def concate_claim_evidence(data):
    print("Concatenating...")
    for x in tqdm(data):
        text = [x['claim']]
        for s in x['evidence']:
            text.append(s[2])
        x['text'] = ' '.join(text)
        x['label'] = label_dict[x['label'].upper()]
    return data


for name in ['bert_train.jsonl', 'bert_test.jsonl']:
    data = []
    with open (name, encoding = 'utf-8') as f:
        for l in tqdm(f.readlines()):
            data.append(json.loads(l))
    data = concate_claim_evidence(data)

    with open ('new-' + name,'w', encoding = 'utf-8') as f:
        f.write('\n'.join([json.dumps(x) for x in data]))