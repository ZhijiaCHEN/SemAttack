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

def generate_embdding(tokenizer, model, data):
    # 产生 word到embedding的映射，每个word在不同的句子下有不同的embedding，所以每个单词对应一个list of embeddings
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
            # 每个单词有个可能被拆分为多个token, ##预示着当前的token属于上一个单词的一部分
            word = tokenized_words[i]
            embed = output[0][0][i].cpu()
            if len(word) > 2 and word[:2] == '##':
                
                last_word += word[2:]
                embeddings.append(embed)
            else:
                if len(embeddings) > 0:
                    # 我们对每个单词取其所有token的embedding的平均值作为该单词的embedding
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
    # 根据输入的参数加载对应模型，获取数据的word embedding
    if args.model_name == 'bert':
        model = Pretrained_Fever_BERT(model_states)
    elif args.model_name == 'kgat':
        model = load_KernelGAT(model_states)
    model = model.to(device)
    model.eval()
    data = load_fever()
    generate_embdding(tokenizer, model, data)