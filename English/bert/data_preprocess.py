import json, os
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import BertTokenizer, BertForSequenceClassification
import torch, joblib, random
from tqdm import tqdm
from util import args, logger
from datasets import Dataset, load_metric
from torch.utils.data import DataLoader
from attack_KernelGAT import load_KernelGAT
from models import Pretrained_Fever_BERT


# def generate_embdding(tokenizer, model, data):
#     word_to_embeddings = {}
    
#     for x in tqdm(data, total=len(data)):
#         indexed_tokens = tokenizer.encode(x,add_special_tokens=False)
#         token_tensor = torch.tensor([indexed_tokens]).to(device)
#         tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
#         with torch.no_grad():
#             output = model(token_tensor, output_hidden_states=True)
#         last_word = ''
#         embeddings = []
#         for i in range(len(indexed_tokens)):
#             word = tokenized_words[i]
#             embed = output['hidden_states'][0][0][i].cpu()
#             if len(word) > 2 and word[:2] == '##':
#                 last_word += word[2:]
#                 embeddings.append(embed)
#             else:
#                 if len(embeddings) > 0:
#                     word_to_embeddings.setdefault(last_word, []).append(torch.mean(torch.stack(embeddings), 0))
#                 last_word = word
#                 embeddings = [embed]

#     word_embedding = []
#     word_list = []
#     for word in sorted(word_to_embeddings.keys()):
#         word_embedding.extend(word_to_embeddings[word])
#         word_list.extend([word] * len(word_to_embeddings[word]))
#     word_embedding = torch.stack(word_embedding)
#     # joblib.dump(data, "fever_data.pkl")
#     joblib.dump(word_list, '/'.join(args.test_data.split('/')[:-1]) + "/word-list.pkl")
#     torch.save(word_embedding, '/'.join(args.test_data.split('/')[:-1]) + "/word-embedding.pt")

def generate_embdding(tokenizer, model, data):
    word_to_embeddings = {}
    
    for x in tqdm(data, total=len(data)):
        indexed_tokens = tokenizer.encode(x, add_special_tokens=True)
        token_tensor = torch.tensor([indexed_tokens]).to(device)
        tokenized_words = [tokenizer._convert_id_to_token(x) for x in indexed_tokens]
        with torch.no_grad():
            output = model.word_embeddings(token_tensor)
        last_word = ''
        embeddings = []
        for i in range(len(indexed_tokens)):
            word = tokenized_words[i]
            embed = output[0][0][i].cpu()
            if len(word) > 2 and word[:2] == '##':
                last_word += word[2:]
                embeddings.append(embed)
            else:
                if len(embeddings) > 0:
                    word_to_embeddings.setdefault(last_word, []).append(torch.mean(torch.stack(embeddings), 0))
                last_word = word
                embeddings = [embed]

    word_embedding = []
    word_list = []
    for word in sorted(word_to_embeddings.keys()):
        word_embedding.extend(word_to_embeddings[word])
        word_list.extend([word] * len(word_to_embeddings[word]))
    word_embedding = torch.stack(word_embedding)
    # joblib.dump(data, "fever_data.pkl")
    joblib.dump(word_list, '/'.join(args.test_data.split('/')[:-1]) + f"/word-list-{args.model_name}.pkl")
    torch.save(word_embedding, '/'.join(args.test_data.split('/')[:-1]) + f"/word-embedding-{args.model_name}.pt")

def load_fever():
    data = []
    with open(args.test_data, encoding='utf-8') as f:
        for l in f.readlines():
            x = json.loads(l)
            data.append(x['claim'])
    return data

if __name__ == "__main__":
    device = torch.device("cuda")
    #generate_embdding()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    device = torch.device("cuda")
    model_states = torch.load(args.model_states)
    if args.model_name == 'bert':
        model = Pretrained_Fever_BERT(model_states)
    elif args.model_name == 'kgat':
        model = load_KernelGAT(model_states)
    model = model.to(device)
    model.eval()
    data = load_fever()
    generate_embdding(tokenizer, model, data)