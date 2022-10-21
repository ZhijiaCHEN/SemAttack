import os
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pytorch_transformers import BertTokenizer

from util import logger, args, set_seed, root_dir
import joblib
import models


class YelpDataset(Dataset):
    def __init__(self, path):
        cache_path = path.split('.')[0] + "-tokenized.pkl"
        self.max_len = 130
        if os.path.exists(cache_path):
            self.data = random.sample(joblib.load(cache_path), 1000)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            self.data = joblib.load(path)
            for data in tqdm(self.data):
                data['seq'] = tokenizer.encode(('[CLS] ' + data['claim']))
                if len(data['seq']) > self.max_len:
                    data['seq'] = data['seq'][:self.max_len]
                data['seq_len'] = len(data['seq'])
                data['seq'].extend([0] * (self.max_len - len(data['seq'])))
                data['seq'] = torch.tensor(data['seq'])
            joblib.dump(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index]['seq'], self.data[index]['seq_len'], self.data[index]['label']) 


def test(test_batch):
    model.eval()
    total = 0.0
    acc = 0.0
    with torch.no_grad():
        for seq, seq_len, label in tqdm(test_batch):
            out = model(seq.to(device), seq_len.to(device), label.to(device))
            logits = out['pred'].detach().cpu()
            pred = logits.argmax(dim=-1)
            num = pred.size(0)
            total += num
            acc += pred.eq(label).sum().item()

    logger.info("ACC: {} {} {}".format(acc / total, acc, total))
    return acc


def train():
    logger.info("Number training instances: {}".format(len(train_data)))
    logger.info("Number test instances: {}".format(len(test_data)))

    train_batch = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_batch = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
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
        # for bi, batch in enumerate(tqdm(train_batch)):
        #     model.zero_grad()
        #     batch['seq'] = torch.stack(batch['seq']).t().to(device)
        #     batch['seq_len'] = batch['seq_len'].to(device)
        #     batch['label'] = batch['label'].to(device)
        for seq, seq_len, label in tqdm(train_batch):
            model.zero_grad()
            # out = model(batch['seq'], batch['seq_len'])
            out = model(seq.to(device), seq_len.to(device), label.to(device))
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

            save = '/'.join(args.test_data.split('/')[:-1] + ['model.pth'])
            logger.info("Saving model to {}".format(save))
            torch.save(model.state_dict(), save)


if __name__ == '__main__':
    device = torch.device("cuda" if args.cuda else "cpu")
    train_data = YelpDataset(args.train_data)
    test_data = YelpDataset(args.test_data)
    set_seed(args)

    model = models.BertC(name='bert-base-cased', dropout=args.dropout, num_class=3)
    model.to(device)
    # if args.load:
    #     model.load_state_dict(torch.load(args.load))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=5e-5)

    # train()
    train_batch = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    train()
