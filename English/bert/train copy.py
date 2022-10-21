import os
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from util import logger, args, set_seed, root_dir
import joblib

class YelpDataset(Dataset):
    def __init__(self, path):
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        label_dict = {'NOT ENOUGH INFO': torch.tensor(0), 'SUPPORTS': torch.tensor(1), 'REFUTES': torch.tensor(2)}
        cache_path = 'tokenized_' + path
        self.max_len = 130
        if os.path.exists(cache_path):
            self.data = joblib.load(cache_path)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.data = joblib.load(path)
            for data in tqdm(self.data):
                encoded_dict = tokenizer.encode_plus(
                    data['raw_text'],                      # Sentence to encode.
                    add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                    max_length = self.max_len,           # Pad & truncate all sentences.
                    # pad_to_max_length = self.max_len,
                    padding = 'max_length',
                    truncation = True,
                    return_attention_mask = True,   # Construct attn. masks.
                    return_tensors = 'pt',     # Return pytorch tensors.
                )
                data['seq'] = encoded_dict['input_ids'].reshape(-1)
                data['attention_mask'] = encoded_dict['attention_mask'].reshape(-1)
                data['label'] = label_dict[data['label'].upper()]
            joblib.dump(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['seq'], self.data[index]['attention_mask'], self.data[index]['label']

# def collate_fn(data):
#     """
#        data: is a list of tuples with (example, label, length)
#              where 'example' is a tensor of arbitrary shape
#              and label/length are scalars
#     """
#     # _, labels, lengths = zip(*data)
#     # max_len = max(lengths)
#     # n_ftrs = data[0][0].size(1)
#     # features = torch.zeros((len(data), max_len, n_ftrs))
#     # labels = torch.tensor(labels)
#     # lengths = torch.tensor(lengths)

#     # for i in range(len(data)):
#     #     j, k = data[i][0].size(0), data[i][0].size(1)
#     #     features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

#     # return features.float(), labels.long(), lengths.long()
#     return {
#         'seq': torch.stack([torch.tensor(x['seq']) for x in data]),
#         'seq_len': torch.stack([torch.tensor(x['seq_len']) for x in data]),
#         'label': torch.stack([torch.tensor(x['label']) for x in data]),
#         'attention_mask': 
#     }

def test(test_batch):
    model.eval()
    total = 0.0
    acc = 0.0
    with torch.no_grad():
        for bi, batch in enumerate(test_batch):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            out = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            logits = out['logits'].detach().cpu()
            pred = logits.argmax(dim=-1)
            num = pred.size(0)
            total += num
            acc += pred.eq(b_labels.cpu()).sum().item()

    logger.info("ACC: {} {} {}".format(acc / total, acc, total))
    return acc


def train():
    logger.info("Number training instances: {}".format(len(train_data)))
    logger.info("Number test instances: {}".format(len(test_data)))

    train_batch = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_batch = DataLoader(test_data, batch_size=1, shuffle=False)
    # test_batch = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    best = 0.

    for epoch in range(args.epochs):
        logger.info("Epoch {} out of {}".format(epoch + 1, args.epochs))
        t1 = time.time()
        model.train()
        train_loss = 0
        tot = 0
        # progress = tqdm(enumerate(train_batch), desc='train loss')
        # for bi, batch in progress:
        for bi, batch in enumerate(tqdm(train_batch)):
            model.zero_grad()
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            # out = model(batch['seq'], batch['seq_len'])
            out = model(input_ids=b_input_ids, 
                        attention_mask=b_input_mask, 
                        labels=b_labels)
            loss = torch.mean(out['loss'])
            train_loss += loss.item()
            tot += 1
            # progress.set_description('train loss: ' + str(train_loss / tot))
            # progress.refresh()
            loss.backward()
            optimizer.step()

        t2 = time.time()
        train_loss = train_loss / tot
        logger.info("time: {} loss: {}".format(t2 - t1, train_loss))

        test_acc = test(test_batch)
        if test_acc > best:
            best = test_acc
            logger.info("new best score: " + str(best))

            save = os.path.join(root_dir, 'model.pth')
            logger.info("Saving model to {}".format(save))
            torch.save(model.state_dict(), save)


if __name__ == '__main__':
    train_data = YelpDataset(args.train_data)
    test_data = YelpDataset(args.test_data)
    set_seed(args)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    # torch.cuda.set_device(1)
    device = torch.device("cuda" if args.cuda else "cpu")
    # device = torch.device("cpu")
    model.to(device)
    # if args.load:
    #     model.load_state_dict(torch.load(args.load))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-6)

    # train()
    # train_batch = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    train()
