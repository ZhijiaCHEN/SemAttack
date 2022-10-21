import random, json, joblib
with open('fever2-fixers-dev.jsonl', encoding='utf-8') as f:
    data = f.readlines()
    random.shuffle(data)
    splitIndex = int(0.9 * len(data))
    train = data[:splitIndex]
    test = data[splitIndex:]

label_dict = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}
with open('fever2-train.jsonl', 'w', encoding='utf-8') as f:
    f.writelines(train)
train = [json.loads(l) for l in train]
train = [{"text": x["claim"].lower(), "label": label_dict[x["label"].upper()]} for x in train]
# with open('fever2-train.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(train, indent=4))
joblib.dump(train, 'train-data.pkl')
    
with open('fever2-test.jsonl', 'w', encoding='utf-8') as f:
    f.writelines(test)
test = [json.loads(l) for l in test]
test = [{"text": x["claim"].lower(), "label": label_dict[x["label"].upper()]} for x in test]
# with open('fever2-test.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(test, indent=4))
joblib.dump(test, 'test-data.pkl')