#%%
import json
import random
import codecs
import joblib
import os
import numpy as np
import torch
from util import logger, root_dir, load_data, args
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy

# from CW_attack_kgat import CarliniL2

import models
from pytorch_transformers import BertTokenizer

import sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from bert_score import BERTScorer
from models import BertForSequenceEncoder
from models import BertForSequenceClassification
import json
from models import inference_model
from KernelGAT_data_loader import KGAT_Dataset, DataLoaderTest

# %%
class BERT_Dataset(Dataset):
    def __init__(self, path, sample, max_len = 512):
        self.label_dict = {'NOT ENOUGH INFO': 0, 'SUPPORTS': 1, 'REFUTES': 2}
        self.max_len = max_len
        self.data = load_data(path)
        for x in self.data:
            if type(x['label']) == str:
                x['label'] = self.label_dict[x['label']]
        if sample > 0:
            self.data = random.sample(self.data, sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
def cal_ppl(text, model, tokenizer):
    assert isinstance(text, list)
    encodings = tokenizer('\n\n'.join(text), return_tensors='pt')
    max_length = 128 #model.config.n_positions
    stride = 128
    lls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to('cuda')
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc).item()

    return ppl


def cal_bert_score(cands, refs, scorer):
    _, _, f1 = scorer.score(cands, refs)
    return f1.mean()


def transform(seq, unk_words_dict=None):
    if unk_words_dict is None:
        unk_words_dict = {}
    if not isinstance(seq, list):
        seq = seq.squeeze().cpu().numpy().tolist()
    unk_count = 0
    for x in seq:
        if x == 100:
            unk_count += 1
    if unk_count == 0 or len(unk_words_dict) == 0:
        return tokenizer.convert_tokens_to_string([tokenizer._convert_id_to_token(x) for x in seq])
    else:
        tokens = []
        for idx, x in enumerate(seq):
            if x == 100 and len(unk_words_dict[idx]) != 0:
                unk_words = unk_words_dict[idx]
                unk_word = random.choice(unk_words)
                tokens.append(unk_word)
            else:
                tokens.append(tokenizer._convert_id_to_token(x))
        return tokenizer.convert_tokens_to_string(tokens)


def difference(a, b):
    tot = 0
    for x, y in zip(a, b):
        if x != y:
            tot += 1

    return tot


def get_similar_dict(similar_dict):
    similar_char_dict = {0: [0], 101: [101], 102: [102]}
    for k, v in similar_dict.items():
        k = tokenizer._convert_token_to_id(k)
        v = [tokenizer._convert_token_to_id(x[0]) for x in v]
        if k not in v:
            v.append(k)
        while 100 in v:
            v.remove(100)
        if len(v) >= 1:
            similar_char_dict[k] = v
        else:
            similar_char_dict[k] = [k]

    return similar_char_dict


def get_knowledge_dict(input_knowledge_dict):
    knowledge_dict = {0: [0], 101: [101], 102: [102]}
    for k, v in input_knowledge_dict.items():
        k = tokenizer._convert_token_to_id(k)
        v = [tokenizer._convert_token_to_id(x[0]) for x in v]
        if k not in v:
            v.append(k)
        while 100 in v:
            v.remove(100)
        if len(v) >= 1:
            knowledge_dict[k] = v
        else:
            knowledge_dict[k] = [k]

    return knowledge_dict


def get_bug_dict(input_bug_dict, input_ids):
    bug_dict = {0: [0], 101: [101], 102: [102]}
    unk_words_dict = {}
    input_ids = input_ids.squeeze().cpu().numpy().tolist()
    unk_cnt = 0
    for x in input_ids:
        if x == 100:
            unk_cnt += 1
    token_list = [tokenizer._convert_id_to_token(x) for x in input_ids]
    for i in range(len(token_list)):
        if input_ids[i] in bug_dict:
            for j in range(len(token_list)):
                if input_ids[i] == input_ids[j]:
                    if j in unk_words_dict:
                        unk_words_dict[i] = unk_words_dict[j]
                    break
            continue
        word = token_list[i]
        if word not in input_bug_dict:
            bug_dict[input_ids[i]] = [input_ids[i]]
            continue
        candidates = input_bug_dict[word]
        unk_id = 100
        unk_list = []
        for unk_word in [x[0] for x in candidates if tokenizer._convert_token_to_id(x[0]) == unk_id]:
            adv_seq = deepcopy(token_list)
            adv_seq[i] = unk_word
            adv_seq = tokenizer.encode(tokenizer.convert_tokens_to_string(adv_seq))
            adv_unk_cnt = 0
            for x in adv_seq:
                if x == 100:
                    adv_unk_cnt += 1
            if adv_unk_cnt == unk_cnt + 1:
                unk_list = [unk_word]
                break
        unk_words_dict[i] = unk_list
        candidates = [tokenizer._convert_token_to_id(x[0]) for x in candidates]
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        if len(unk_list) == 0:
            while 100 in candidates:
                candidates.remove(100)
        if input_ids[i] not in candidates:
            candidates.append(input_ids[i])
        bug_dict[input_ids[i]] = candidates

    return bug_dict, unk_words_dict

#%%
def check_consistency():
    logger.info("Start checking consistency")

    adv_text = joblib.load(os.path.join(root_dir, 'adv_text.pkl'))
    for i in adv_text:
        i['adv_text'] = i['adv_text'].replace('[CLS] ', '')

    adv_text = YelpDataset(adv_text, raw=True)
    test_batch = DataLoader(adv_text, batch_size=args.batch_size, shuffle=False)

    inconsistent = []
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(test_batch)):
            batch['seq'] = torch.stack(batch['seq']).t().to(device)
            batch['seq_len'] = batch['seq_len'].to(device)
            out = model(batch['seq'], batch['seq_len'])
            logits = out['pred'].detach().cpu()
            pred = logits.argmax(dim=-1)
            if pred[0].item() != batch['pred'][0]:
                inconsistent.append((bi, batch))

    logger.info("Num of inconsistent: {}".format(len(inconsistent)))
    if len(inconsistent) != 0:
        joblib.dump(inconsistent, os.path.join(root_dir, 'inconsistent_adv.pkl'))

    return adv_text


def validate(model):
    logger.info("Start validation")

    adv_text = joblib.load(os.path.join(root_dir, 'adv_text.pkl'))
    for i in adv_text:
        i['adv_text'] = i['adv_text'].replace('[CLS] ', '')

    adv_text_dataset = YelpDataset(adv_text, raw=True)
    test_batch = DataLoader(adv_text_dataset, batch_size=args.batch_size, shuffle=False)

    # states = torch.load(args.load, map_location=torch.device('cuda'))
    # states['proj.weight'] = states.pop('classifier.weight')
    # states['proj.bias'] = states.pop('classifier.bias')
    # states.pop('bert.embeddings.position_ids')
    
    # test_model = models.BertC(dropout=args.dropout, num_class=3)
    # test_model.load_state_dict(states)
    # test_model = test_model.to(device)
    # test_model.eval()
    
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(test_batch)):
            batch['seq'] = torch.stack(batch['seq']).t().to(device)
            batch['seq_len'] = batch['seq_len'].to(device)
            out = model(batch['seq'], batch['seq_len'])
            logits = out['pred'].detach().cpu()
            pred = logits.argmax(dim=-1)
            adv_text[bi]['pred_validated'] = pred[0].item()
    del model
    # del test_model
    torch.cuda.empty_cache()
    joblib.dump(adv_text, os.path.join(root_dir, 'adv_text_validated.pkl'))

    acc = 0
    origin_mistake = 0
    total = 0
    total_change = 0
    total_word = 0
    orig_text_eval = []
    adv_text_eval = []
    for item in tqdm(adv_text):
        if item['ori_pred'] != item['label']:
            origin_mistake += 1
            continue
        if args.untargeted and item['pred_validated'] != item['label'] or not args.untargeted and item['pred_validated'] == item['target']:
            acc += 1
            total_change += item['diff']
            total_word += item['seq_len']
            orig_text_eval.append(item['orig_text'].replace('[CLS]', '').strip())
            adv_text_eval.append(item['adv_text'].replace('[CLS]', '').strip())
        total += 1

    model_id = 'gpt2-medium'
    ppl_model = GPT2LMHeadModel.from_pretrained(model_id).to('cuda')
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    bs_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    orig_ppl = cal_ppl(orig_text_eval, ppl_model, ppl_tokenizer)
    adv_ppl = cal_ppl(adv_text_eval, ppl_model, ppl_tokenizer)
    bs = cal_bert_score(adv_text_eval, orig_text_eval, bs_scorer)

    suc = float(acc / total) * 100
    change_rate = float(total_change / total_word) * 100
    origin_acc = (1 - origin_mistake / len(adv_text)) * 100

    logger.info(sys.argv)
    logger.info('orig acc：{:.1f}%'.format(origin_acc))
    logger.info('attack success：{:.1f}'.format(acc))
    logger.info('orig pred success：{:.1f}'.format(total))
    logger.info('success rate：{:.1f}%'.format(suc))
    logger.info('change rate: {:.1f}%'.format(change_rate))
    logger.info('original ppl: {:.1f}'.format(orig_ppl))
    logger.info('adv ppl: {:.1f}'.format(adv_ppl))
    logger.info('bert score: {:.3f}'.format(bs))

# %%
def check_model(model, data_val):
    model.eval()
    device = next(model.parameters()).device
    test_batch = DataLoader(data_val, batch_size=args.batch_size, shuffle=False)
    correct = mistake = 0
    for batch_index, batch in enumerate(tqdm(test_batch)):
        #kgat_inputs, batch, label, ids = data
        input, mask, segment = batch['kgat input']
        inp_tensor = torch.tensor(input).to(device)
        msk_tensor = torch.tensor(mask).to(device)
        seg_tensor = torch.tensor(segment).to(device)
        
        logits = model(inp_tensor, msk_tensor, seg_tensor)
        #logits = model((inp_tensor, msk_tensor, seg_tensor))
        # test original acc

        prediction = torch.max(logits, 1)[1]
        ori_prediction = prediction
        # print(f"label={label[0].item()}, predict={ori_prediction[0].item()}")
        if ori_prediction[0].item() != batch['label'][0]:
            #continue
            mistake += 1
        else:
            correct += 1
        # batch['orig_correct'] = torch.sum((prediction == label).float())
    acc = correct/(correct + mistake)
    print(f"{correct=}, {mistake=}, {acc=}")

def check_model2(model, test_batch):
    model.eval()
    device = next(model.parameters()).device
    correct = mistake = 0
    for batch_index, batch in enumerate(tqdm(test_batch)):
        #kgat_inputs, batch, label, ids = data
        inputs, labels, ids = batch
        #logits = model(inp_tensor, msk_tensor, seg_tensor)
        logits = model(inputs)
        # test original acc

        prediction = torch.max(logits, 1)[1]
        ori_prediction = prediction
        # print(f"label={label[0].item()}, predict={ori_prediction[0].item()}")
        if ori_prediction[0].item() != labels[0]:
            #continue
            mistake += 1
        else:
            correct += 1
        # batch['orig_correct'] = torch.sum((prediction == label).float())
    acc = correct/(correct + mistake)
    print(f"{correct=}, {mistake=}, {acc=}")

def load_KernelGAT(state_dict):
    bert_model = BertForSequenceEncoder.from_pretrained(args.bert_pretrain)
    model = inference_model(bert_model, args)
    model.load_state_dict(state_dict['model'])
    return model

#%%
logger.info("Start attack")
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
device = torch.device("cuda")
#
if args.model_name == 'bert':
    # model = models.Pretrained_Fever_BERT(model_states)
    model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
    #model_states.pop('bert.embeddings.position_ids')
    #model.load_state_dict(model_states)
    
    test_data = BERT_Dataset(args.test_data, args.sample)
elif args.model_name == 'kgat':
    label_map = {'SUPPORTS': 0, 'REFUTES': 1, 'NOT ENOUGH INFO': 2}
    test_data = KGAT_Dataset(args.test_data, label_map, tokenizer, sampleCnt = args.sample)
    # test_data2 = DataLoaderTest(args.test_data, label_map=label_map, tokenizer=tokenizer, args=args, batch_size=1)
    model_states = torch.load(args.model_states)
    model = load_KernelGAT(model_states).to(device)
# model.eval()

# %%
check_model(model, test_data)


#%%
args.lr = 0.1
args.max_steps = 100
args.debug_cw = True
args.clip = 0.5
args.decreasing_temp = False
args.temp = 1e-1
args.sample = 1000
# cw_word_attack(test_data, model)
# validate(model)

# %%
from torch import optim
class CarliniL2:

    def __init__(self, targeted=True, search_steps=None, max_steps=None, cuda=False, debug=False, num_classes=3):
        logger.info(("const confidence lr:", args.const, args.confidence, args.lr))
        self.debug = debug
        self.targeted = targeted
        self.num_classes = num_classes
        self.confidence = args.confidence  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        self.initial_const = args.const  # bumped up from default of .01 in reference code
        self.binary_search_steps = search_steps or 1
        self.repeat = self.binary_search_steps >= 10
        self.max_steps = max_steps or args.max_steps
        self.abort_early = True
        self.cuda = cuda
        self.mask = None
        self.batch_info = None
        self.wv = None
        self.claim_seq = None
        self.seq_len = None
        self.kgat_inputs = None
        self.kgat_mask= None
        self.kgat_segment = None
        self.init_rand = False  # an experiment, does a random starting point help?

    def _compare(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            # if self.targeted:
            #     output[target] -= self.confidence
            # else:
            #     output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target
        else:
            return output != target

    def _compare_untargeted(self, output, target):
        if not isinstance(output, (float, int, np.int64)):
            output = np.copy(output)
            # if self.targeted:
            #     output[target] -= self.confidence
            # else:
            #     output[target] += self.confidence
            output = np.argmax(output)
        if self.targeted:
            return output == target + 1 or output == target - 1
        else:
            return output != target

    def _loss(self, output, target, dist, scale_const):
        # compute the probability of the label class versus the maximum other
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            # if targeted, optimize for making the other class most likely
            loss1 = torch.clamp(other - real + self.confidence, min=0.)  # equiv to max(..., 0.)
        else:
            # if non-targeted, optimize for making this class least likely.
            loss1 = torch.clamp(real - other + self.confidence, min=0.)  # equiv to max(..., 0.)
        loss1 = torch.sum(scale_const * loss1)
        loss2 = dist.sum()
        if args.debug_cw:
            print("loss 1:", loss1.item(), "   loss 2:", loss2.item())
        loss = loss1 + loss2
        return loss

    def _optimize(self, optimizer, model, input_var, modifier_var, target_var, scale_const_var, input_token=None):
        # apply modifier and clamp resulting image to keep bounded from clip_min to clip_max

        batch_adv_sent = []
        if self.mask is None:
            # not word-level attack
            input_adv = modifier_var + input_var
            output = model(input_adv)
            input_adv = model.get_embedding()
            input_var = input_token
            seqback = model.get_seqback()
            batch_adv_sent = seqback.adv_sent.copy()
            seqback.adv_sent = []
            # input_adv = self.itereated_var = modifier_var + self.itereated_var
        else:
            # word level attack
            input_adv = modifier_var * self.mask + self.itereated_var
            # input_adv = modifier_var * self.mask + input_var
            for i in range(input_adv.size(0)):
                # for batch size
                new_word_list = []
                add_start = self.batch_info['attack_start'][i]
                add_end = self.batch_info['attack_end'][i]
                for j in range(add_start, add_end):
                    # print(self.wv[self.claim_seq[0][j].item()])
                    # if self.claim_seq[0][j].item() not in self.wv.keys():

                    similar_wv = model.pred_model.bert.embeddings.word_embeddings(torch.LongTensor(self.wv[self.claim_seq[i][j].item()]).cuda())
                    new_placeholder = input_adv[i, j].data
                    temp_place = new_placeholder.expand_as(similar_wv)
                    new_dist = torch.norm(temp_place - similar_wv.data, 2, -1)  # 2范数距离，一个字一个float
                    _, new_word = torch.min(new_dist, 0)
                    new_word_list.append(new_word.item())
                    # input_adv.data[j, i] = self.wv[new_word.item()].data
                    input_adv.data[i, j] = self.itereated_var.data[i, j] = similar_wv[new_word.item()].data
                    del temp_place
                batch_adv_sent.append(new_word_list)

            perturbed = model.pred_model.bert.embeddings.word_embeddings(self.kgat_inputs)
            perturbed[:,:self.seq_len,:] = input_adv.repeat(model.evi_num, 1, 1)
            output = model(self.kgat_inputs, self.kgat_mask, self.kgat_segment, perturbed=perturbed)
            if args.debug_cw:
                print("output:", batch_adv_sent)
                print("input_adv:", input_adv)
                print("output:", output)
                adv_claim_seq = torch.tensor(self.claim_seq)
                for bi, (add_start, add_end) in enumerate(zip(self.batch_info['attack_start'], self.batch_info['attack_end'])):
                    adv_claim_seq.data[bi, add_start:add_end] = torch.LongTensor(batch_adv_sent)
                print("out:", adv_claim_seq)
                print("out embedding:", model.pred_model.bert.embeddings.word_embeddings(adv_claim_seq))
                
                adv_seq = self.kgat_inputs.clone()
                adv_seq[:, :self.seq_len] = adv_claim_seq.repeat(args.evi_num, 1)
                out = model(adv_seq, self.kgat_mask, self.kgat_segment)
                print("out:", out)

        def reduce_sum(x, keepdim=True):
            # silly PyTorch, when will you get proper reducing sums/means?
            for a in reversed(range(1, x.dim())):
                x = x.sum(a, keepdim=keepdim)
            return x

        def l1_dist(x, y, keepdim=True):
            d = torch.abs(x - y)
            return reduce_sum(d, keepdim=keepdim)

        def l2_dist(x, y, keepdim=True):
            d = (x - y) ** 2
            return reduce_sum(d, keepdim=keepdim)

        # distance to the original input data
        if args.l1:
            dist = l1_dist(input_adv, input_var, keepdim=False)
        else:
            dist = l2_dist(input_adv, input_var, keepdim=False)
        loss = self._loss(output, target_var, dist, scale_const_var)
        if args.debug_cw:
            print(loss)
        optimizer.zero_grad()
        if input_token is None:
            loss.backward()
        else:
            loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_([modifier_var], args.clip)
        optimizer.step()
        # modifier_var.data -= 2 * modifier_var.grad.data
        # modifier_var.grad.data.zero_()

        loss_np = loss.item()
        dist_np = dist.data.cpu().numpy()
        output_np = output.data.cpu().numpy()
        input_adv_np = input_adv.data.cpu().numpy()
        return loss_np, dist_np, output_np, input_adv_np, batch_adv_sent

    def run(self, model, input, target, batch_idx=0, batch_size=None, input_token=None):
        if batch_size is None:
            batch_size = input.size(0)  # ([length, batch_size, nhim])
        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        scale_const = np.ones(batch_size) * self.initial_const
        upper_bound = np.ones(batch_size) * 1e10

        # python/numpy placeholders for the overall best l2, label score, and adversarial image
        o_best_l2 = [1e10] * batch_size
        o_best_score = [-1] * batch_size
        o_best_logits = {}
        if input_token is None:
            best_attack = input.cpu().detach().numpy()
            o_best_attack = input.cpu().detach().numpy()
        else:
            best_attack = input_token.cpu().detach().numpy()
            o_best_attack = input_token.cpu().detach().numpy()
        self.o_best_sent = {}
        self.best_sent = {}

        # setup input (image) variable, clamp/scale as necessary
        input_var = torch.tensor(input, requires_grad=False)
        self.itereated_var = torch.tensor(input_var)
        # setup the target variable, we need it to be in one-hot form for the loss function
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if self.cuda:
            target_onehot = target_onehot.cuda()
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = torch.tensor(target_onehot, requires_grad=False)

        # setup the modifier variable, this is the variable we are optimizing over
        modifier = torch.zeros(input_var.size()).float()
        if self.cuda:
            modifier = modifier.cuda()
        modifier_var = torch.tensor(modifier, requires_grad=True)

        optimizer = optim.Adam([modifier_var], lr=args.lr)

        for search_step in range(self.binary_search_steps):
            if args.debug_cw:
                print('Batch: {0:>3}, search step: {1}'.format(batch_idx, search_step))
                print('Const:')
                for i, x in enumerate(scale_const):
                    print(i, x)
            best_l2 = [1e10] * batch_size
            best_score = [-1] * batch_size
            best_logits = {}
            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat and search_step == self.binary_search_steps - 1:
                scale_const = upper_bound

            scale_const_tensor = torch.from_numpy(scale_const).float()
            if self.cuda:
                scale_const_tensor = scale_const_tensor.cuda()
            scale_const_var = torch.tensor(scale_const_tensor, requires_grad=False)

            for step in range(self.max_steps):
                # perform the attack
                if self.mask is None:
                    if args.decreasing_temp:
                        cur_temp = args.temp - (args.temp - 0.1) / (self.max_steps - 1) * step
                        model.set_temp(cur_temp)
                        if args.debug_cw:
                            print("temp:", cur_temp)
                    else:
                        model.set_temp(args.temp)
                # output 是攻击后的model的test输出  adv_img是输出的词向量矩阵， adv_sents是字的下标组成的list
                loss, dist, output, adv_img, adv_sents = self._optimize(
                    optimizer,
                    model,
                    input_var,
                    modifier_var,
                    target_var,
                    scale_const_var,
                    input_token)

                for i in range(batch_size):
                    target_label = target[i]
                    output_logits = output[i]
                    output_label = np.argmax(output_logits)
                    di = dist[i]
                    if self.debug:
                        if step % 100 == 0:
                            print('{0:>2} dist: {1:.5f}, output: {2:>3}, {3:5.3}, target {4:>3}'.format(
                                i, di, output_label, output_logits[output_label], target_label))
                    if di < best_l2[i] and self._compare_untargeted(output_logits, target_label):
                        # if self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best step,  prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                i, best_l2[i], di))
                        best_l2[i] = di
                        best_score[i] = output_label
                        best_logits[i] = output_logits
                        best_attack[i] = adv_img[i]
                        self.best_sent[i] = adv_sents[i]
                    if di < o_best_l2[i] and self._compare(output_logits, target_label):
                        # if self._compare(output_logits, target_label):
                        if self.debug:
                            print('{0:>2} best total, prev dist: {1:.5f}, new dist: {2:.5f}'.format(
                                i, o_best_l2[i], di))
                        o_best_l2[i] = di
                        o_best_score[i] = output_label
                        o_best_logits[i] = output_logits
                        o_best_attack[i] = adv_img[i]
                        self.o_best_sent[i] = adv_sents[i]
                sys.stdout.flush()
                # end inner step loop

            # adjust the constants
            batch_failure = 0
            batch_success = 0
            for i in range(batch_size):
                if self._compare(o_best_score[i], target[i]) and o_best_score[i] != -1:
                    batch_success += 1
                    if args.debug_cw:
                        print(self.o_best_sent[i])
                        print(o_best_score[i])
                        print(o_best_logits[i])
                elif self._compare_untargeted(best_score[i], target[i]) and best_score[i] != -1:
                    o_best_l2[i] = best_l2[i]
                    o_best_score[i] = best_score[i]
                    o_best_attack[i] = best_attack[i]
                    self.o_best_sent[i] = self.best_sent[i]
                    if args.debug_cw:
                        print(self.o_best_sent[i])
                        print(o_best_score[i])
                        print(o_best_logits[i])
                    batch_success += 1
                else:
                    batch_failure += 1
            # print('Num failures: {0:2d}, num successes: {1:2d}\n'.format(batch_failure, batch_success))
            sys.stdout.flush()
            # end outer search loop

        return o_best_attack


# %%
adv_correct = 0
targeted_success = 0
untargeted_success = 0
orig_correct = 0
tot = 0
tot_diff = 0
tot_len = 0
adv_pickle = []

test_batch = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
cw = CarliniL2(debug=False, targeted=not args.untargeted, cuda=True)
for batch_index, batch in enumerate(tqdm(test_batch)):
    batch_add_start = batch['attack_start']
    batch_add_end = batch['attack_end']

    data = batch['seq'] = torch.stack(batch['seq']).t().to(device)
    seq_len = batch['seq_len'] = batch['seq_len'].to(device)
    if args.untargeted:
        attack_targets = batch['label']
    else:
        if args.strategy == 0:
            if batch['label'][0] == 1:
                attack_targets = torch.full_like(batch['label'], 0)
            else:
                attack_targets = torch.full_like(batch['label'], 1)
        elif args.strategy == 1:
            if batch['label'][0] < 2:
                attack_targets = torch.full_like(batch['label'], 2)
            else:
                attack_targets = torch.full_like(batch['label'], 0)
    label = batch['label'] = batch['label'].to(device)
    attack_targets = attack_targets.to(device)

    # test original acc
    input, mask, segment = batch['kgat input']
    inp_tensor = torch.tensor(input).to(device)
    msk_tensor = torch.tensor(mask).to(device)
    seg_tensor = torch.tensor(segment).to(device)
    out = model(inp_tensor, msk_tensor, seg_tensor)
    prediction = torch.max(out, 1)[1]
    ori_prediction = prediction
    if ori_prediction[0].item() != label[0].item():
        continue
    batch['orig_correct'] = torch.sum((prediction == label).float())

    # prepare attack
    input_embedding = model.pred_model.bert.embeddings.word_embeddings(data)
    cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
    cw_mask = torch.from_numpy(cw_mask).float().to(device)
    for i, seq in enumerate(batch['seq_len']):
        cw_mask[i][1:seq] = 1

    if args.function == 'all':
        cluster_char_dict = get_similar_dict(batch['similar_dict'])
        bug_char_dict, unk_words_dict = get_bug_dict(batch['bug_dict'], batch['seq'][0])
        similar_char_dict = get_knowledge_dict(batch['knowledge_dict'])

        for k, v in cluster_char_dict.items():
            synset = list(set(v + similar_char_dict[k]))
            while 100 in synset:
                synset.remove(100)
            if len(synset) >= 1:
                similar_char_dict[k] = synset
            else:
                similar_char_dict[k] = [k]

        for k, v in bug_char_dict.items():
            synset = list(set(v + similar_char_dict[k]))
            # while 100 in synset:
            #     synset.remove(100)
            if len(synset) >= 1:
                similar_char_dict[k] = synset
            else:
                similar_char_dict[k] = [k]

        all_dict = similar_char_dict
    elif args.function == 'typo':
        all_dict, unk_words_dict = get_bug_dict(batch['bug_dict'], batch['seq'][0])
    elif args.function == 'knowledge':
        all_dict = get_knowledge_dict(batch['knowledge_dict'])
        unk_words_dict = None
    elif args.function == 'cluster':
        all_dict = get_similar_dict(batch['similar_dict'])
        unk_words_dict = None
    else:
        raise Exception('Unknown perturbation function.')
    cw.wv = all_dict
    cw.mask = cw_mask
    cw.claim_seq = data
    cw.batch_info = batch
    cw.seq_len = seq_len
    cw.kgat_inputs = inp_tensor
    cw.kgat_mask = msk_tensor
    cw.kgat_segment = seg_tensor

    # attack
    adv_data = cw.run(model, input_embedding, attack_targets)
    # retest
    adv_claim_seq = torch.tensor(batch['seq']).to(device)
    for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
        if bi in cw.o_best_sent:
            for i in range(add_start, add_end):
                adv_claim_seq.data[bi, i] = all_dict[adv_claim_seq.data[bi, i].item()][cw.o_best_sent[bi][i - add_start]]

    adv_seq = inp_tensor.clone()
    adv_seq[:, :seq_len] = adv_claim_seq.repeat(args.evi_num, 1)
    out = model(adv_seq, msk_tensor, seg_tensor)
    prediction = torch.max(out, 1)[1]
    orig_correct += batch['orig_correct'].item()
    adv_correct += torch.sum((prediction == label).float()).item()
    targeted_success += torch.sum((prediction == attack_targets).float()).item()
    untargeted_success += torch.sum((prediction != label).float()).item()
    tot += len(batch['label'])

    for i in range(len(batch['label'])):
        diff = difference(adv_claim_seq[i], data[i])
        adv_pickle.append({
            'index': batch_index,
            'adv_text': transform(adv_claim_seq[i], unk_words_dict),
            'orig_text': transform(batch['seq'][i]),
            'raw_text': batch['claim'][i],
            'label': label[i].item(),
            'target': attack_targets[i].item(),
            'ori_pred': ori_prediction[i].item(),
            'pred': prediction[i].item(),
            'diff': diff,
            'orig_seq': batch['seq'][i].cpu().numpy().tolist(),
            'adv_seq': adv_claim_seq[i].cpu().numpy().tolist(),
            'seq_len': batch['seq_len'][i].item()
        })
        # assert ori_prediction[i].item() == label[i].item()
        if (args.untargeted and prediction[i].item() != label[i].item()) or (not args.untargeted and prediction[i].item() == attack_targets[i].item()):
            tot_diff += diff
            tot_len += batch['seq_len'][i].item()
            if batch_index % 100 == 0:
                try:
                    # logger.info(("label:", label[i].item()))
                    # logger.info(("pred:", prediction[i].item()))
                    # logger.info(("ori_pred:", ori_prediction[i].item()))
                    # logger.info(("target:", attack_targets[i].item()))
                    # logger.info(("orig:", transform(batch['seq'][i])))
                    # logger.info(("adv:", transform(adv_seq[i], unk_words_dict)))
                    # logger.info(("seq_len:", batch['seq_len'][i].item()))

                    logger.info(("tot:", tot))
                    logger.info(("avg_seq_len: {:.1f}".format(tot_len / tot)))
                    logger.info(("avg_diff: {:.1f}".format(tot_diff / tot)))
                    logger.info(("avg_diff_rate: {:.1f}%".format(tot_diff / tot_len * 100)))
                    logger.info(("orig_correct: {:.1f}%".format(orig_correct / tot * 100)))
                    logger.info(("adv_correct: {:.1f}%".format(adv_correct / tot * 100)))
                    if args.untargeted:
                        logger.info(("targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
                        logger.info(("*untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
                    else:
                        logger.info(("*targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
                        logger.info(("untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
                except:
                    continue
joblib.dump(adv_pickle, os.path.join(root_dir, 'adv_text.pkl'))
logger.info(("tot:", tot))
logger.info(("avg_seq_len: {:.1f}".format(tot_len / tot)))
logger.info(("avg_diff: {:.1f}".format(tot_diff / tot)))
logger.info(("avg_diff_rate: {:.1f}%".format(tot_diff / tot_len * 100)))
logger.info(("orig_correct: {:.1f}%".format(orig_correct / len(test_data) * 100)))
logger.info(("adv_correct: {:.1f}%".format(adv_correct / tot * 100)))
if args.untargeted:
    logger.info(("targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
    logger.info(("*untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
else:
    logger.info(("*targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
    logger.info(("untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
logger.info(("const confidence:", args.const, args.confidence))
# %%
