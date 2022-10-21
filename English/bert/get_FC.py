import joblib
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from util import logger, root_dir, args
from transformers import BertTokenizer

from collections import Counter
from attack_KernelGAT import load_KernelGAT
from models import Pretrained_Fever_BERT

# class YelpDataset(Dataset):
#     def __init__(self, path):
#         cache_path = path + '.FC-cache'
#         self.max_len = 128
#         if os.path.exists(cache_path):
#             self.data = joblib.load(cache_path)
#         else:
#             tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#             self.data = joblib.load(path)
#             clustered_data = []
#             for i, data in enumerate(tqdm(self.data)):
#                 data['seq'] = tokenizer.encode(('[CLS] ' + data['text']))
#                 if len(data['seq']) > self.max_len:
#                     data['seq'] = data['seq'][:self.max_len]
#                 data['seq_len'] = len(data['seq'])
#                 data['similar_dict'] = get_similar_dict(data['seq'])
#                 clustered_data.append(data)
#                 if i % 100 == 0:
#                     joblib.dump(clustered_data, cache_path)
#             joblib.dump(self.data, cache_path)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]

device = torch.device("cuda")
torch.manual_seed(args.seed)

def get_knn(t, k):
    dist = torch.norm(embedding_space - t, dim=1, p=None)
    knn = dist.topk(k, largest=False)
    words = []
    for index in knn.indices:
        words.append(word_list[index])
    count = Counter(words)
    sorted_words = sorted(count.items(), key=lambda x: x[1], reverse=True)
    return sorted_words


def filter_words(words, neighbors):
    words = [item[0] for item in words if item[1] >= neighbors]
    return words


def get_similar_dict(indexed_tokens, cluster_model, tokenizer):
    similar_char_dict = {}
    token_tensor = torch.tensor([indexed_tokens]).to(device)
    #mask_tensor = torch.tensor([[1] * len(indexed_tokens)]).to(device)
    with torch.no_grad():
        encoded_layers, _ = cluster_model(token_tensor, None, None)
    tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
    for i in range(1, len(indexed_tokens)):
        if tokenized_words[i] in word_list:
            words = get_knn(encoded_layers[0][i].cpu(), 700)
            words = filter_words(words, 8)
        else:
            words = []
        if len(words) >= 1:
            similar_char_dict[tokenized_words[i]] = words
        else:
            similar_char_dict[tokenized_words[i]] = [tokenized_words[i]]

    return similar_char_dict

def process_data(data, text_key, cluster_model, tokenizer, max_len):
    cache_path = args.test_data + f'{cluster_model}' + '.FC-cache'
    clustered_data = []
    for i, data in enumerate(tqdm(data)):
        data['seq'] = tokenizer.encode(data[text_key], add_special_tokens=True)
        if len(data['seq']) > max_len:
            data['seq'] = data['seq'][:max_len]
        data['seq_len'] = len(data['seq'])
        data['similar_dict'] = get_similar_dict(data['seq'], cluster_model, tokenizer)
        clustered_data.append(data)
        if i % 100 == 0:
            joblib.dump(clustered_data, cache_path)
    joblib.dump(clustered_data, cache_path)

def process_BERT():
    state_dict = torch.load(args.bert_pretrain)
    model = Pretrained_Fever_BERT(state_dict)
    model.eval()
    model = model.to(device)
    max_len = 130
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    data = joblib.load(args.test_data)
    process_data(data, 'claim', model, tokenizer, max_len)

def process_KernelGAT():
    model = load_KernelGAT(device)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    data = joblib.load(args.test_data)
    max_len = 130
    process_data(data, 'claim', model.pred_model, tokenizer, max_len)

if __name__ == '__main__':
    embedding_space = torch.load(args.embedding_data)
    word_list = joblib.load(args.word_list)
    process_BERT()

