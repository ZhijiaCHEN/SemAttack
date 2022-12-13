# %%
from datasets import load_dataset, load_metric, Dataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from util import args, logger
import torch
import json, random

# %%
# Load data. The data is in jsonl format where each line is a json.
data = []
with open(args.train_data, encoding='utf-8') as f:
    for l in f.readlines():
        data.append(json.loads(l))

# Construct the training data as a list of key-value pairs (dictionary). The "text" field is the input data and it consists of claim and the corresponding evidence. 
# The evidence field is an array of evidences. Each evidence is also an array and the evidence text is the third element, which is all we need from the evidence.
train = [{"labels": x["label"], "text": x["claim"] + " ".join([i[2] for i in x["evidence"]])} for x in data]

# Construct the testing data
data = []
with open(args.test_data, encoding='utf-8') as f:
    for l in f.readlines():
        data.append(json.loads(l))
test = [{"labels": x["label"], "text": x["claim"] + " ".join([i[2] for i in x["evidence"]])} for x in data]

# If the argument for the number of samples is given, we will randomly select the samples. Can be useful when debugging with small portion of data. 
if args.sample > 0:
    train = random.sample(train, args.sample)
    test = random.sample(test, args.sample)

dataset_train = Dataset.from_list(train)
dataset_test = Dataset.from_list(test)

# %%
# Load bert sentence tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Text preprocessing using the bert tokenizer. We apply truncation or padding to make each tokenized sentence have the same length of 256.
def tokenize_function(batch):
    return tokenizer(batch["text"], padding='max_length', truncation=True, max_length=256)

# For convenience, we use the datasets library from the Hugging Face to construct datasets.
# The "text" field will be parsed into a sequence of token ids and stored in the "input_ids" field.
tokenized_train = dataset_train.map(tokenize_function, batched=True)
tokenized_test = dataset_test.map(tokenize_function, batched=True)

# The dataset can be easily set to pytorch / tensorflow tensor format
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# %%
# Construct dataloader, we only need to shuffle the training dataset.
batch_size = args.batch_size
dataloader_train = DataLoader(tokenized_train, shuffle=True, batch_size=batch_size)
dataloader_test = DataLoader(tokenized_test, batch_size=batch_size)

# %%
# We will load a pre-trained Bert model with a sequence classification head. The transformer library has packaged Bert for different types of NLP tasks. Note that the name of the base bert model and the tokenizer are both "bert-base-cased". They must match.
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)

# Set optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = args.epochs
num_training_steps = num_epochs * len(dataloader_train)

# Use a learning rate schedule that helps to change the learning rate during the training process.
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Use CUDA supported GPU if available. 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Construct a progress bar to monitor the training progress.
progress_bar = tqdm(range(num_training_steps))

# We use label accuracy to measure the model performance.
metric = load_metric("accuracy")

# best accuracy
best = 0

# We only need labels, input_ids and attention mask from the tokenized dataset.
input_keys = ['labels', 'input_ids', 'attention_mask']
for epoch in range(num_epochs):
    model.train()
    
    # Training
    for batch in dataloader_train:
        batch = {k: v.to(device) for k, v in batch.items() if k in input_keys}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    
    # Testing
    model.eval()
    for batch in dataloader_test:
        batch = {k: v.to(device) for k, v in batch.items() if k in input_keys}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    acc = metric.compute()['accuracy']
    if acc > best:
        # A better accuracy is found, we save current model.
        best = acc
        logger.info(f"\nBetter accuracy={acc} found.")
        torch.save(model.state_dict(), args.save_path)

# %%
