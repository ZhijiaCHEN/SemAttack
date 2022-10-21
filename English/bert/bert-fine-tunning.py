# %%
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from util import args, logger
import torch
import json, joblib, random

# %%
label_dict = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}

data = []
with open(args.train_data, encoding='utf-8') as f:
    for l in f.readlines():
        data.append(json.loads(l))

train = [{"label": x["label"], "claim": x["claim"] + " " + x["support"]} for x in data]
#train = [{"label": x["label"], "claim": x["claim"]} for x in data]

data = []
with open(args.test_data, encoding='utf-8') as f:
    for l in f.readlines():
        data.append(json.loads(l))
test = [{"label": x["label"], "claim": x["claim"] + " " + x["support"]} for x in data]
#test = [{"label": x["label"], "claim": x["claim"]} for x in data]
if args.sample > 0:
    train = random.sample(train, args.sample)
    test = random.sample(test, args.sample)

dataset_train = Dataset.from_list(train)
dataset_test = Dataset.from_list(test)

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
def tokenize_function(batch):
    return tokenizer(batch["claim"], padding='max_length', truncation=True, max_length=256)

tokenized_train = dataset_train.map(tokenize_function, batched=True).remove_columns(["claim"]).rename_column("label", "labels")
tokenized_test = dataset_test.map(tokenize_function, batched=True).remove_columns(["claim"]).rename_column("label", "labels")
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# %%
batch_size = args.batch_size
dataloader_train = DataLoader(tokenized_train, shuffle=True, batch_size=batch_size)
dataloader_test = DataLoader(tokenized_test, batch_size=batch_size)

# %%
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = args.epochs
num_training_steps = num_epochs * len(dataloader_train)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))
metric = load_metric("accuracy")
best = 0
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader_train:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
    model.eval()
    for batch in dataloader_test:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    acc = metric.compute()['accuracy']
    if acc > best:
        best = acc
        logger.info(f"Better accuracy={acc} found.")
        torch.save(model.state_dict(), args.save_path)

# %%
