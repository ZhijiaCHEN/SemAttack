import random, json, joblib

with open('paper_dev.jsonl', encoding='utf-8') as f:
    dev = f.readlines()

with open('paper_test.jsonl', encoding='utf-8') as f:
    test = f.readlines()

label_dict = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}
test = [json.loads(l) for l in dev + test]
test = [{"claim": x["claim"], "label": label_dict[x["label"].upper()]} for x in test]
joblib.dump(test, 'fever-test.pkl')

with open('train.jsonl', encoding='utf-8') as f:
    train = f.readlines()
# with open('fever-train.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(train, indent=4))
    
train = [json.loads(l) for l in train]
train = [{"claim": x["claim"], "label": label_dict[x["label"].upper()]} for x in train]
joblib.dump(train, 'fever-train.pkl')
# with open('fever-test.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(test, indent=4))