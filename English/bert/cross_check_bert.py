# %%
import json
import random
import codecs
import joblib
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from copy import deepcopy
from torch import optim
from util import logger, load_data, args
import models
BertForSequenceClassification = models.BertForSequenceClassification
inference_model = models.inference_model 
from pytorch_transformers import BertTokenizer
import sys
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from bert_score import BERTScorer
import json

# %%
class YelpDataset(Dataset):
    def __init__(self, path_or_raw, raw=False):
        self.raw = raw
        if not self.raw:
            self.data = joblib.load(path_or_raw)
        else:
            self.max_len = 512
            self.data = path_or_raw
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            for data in self.data:
                data['seq'] = tokenizer.encode(('[CLS] ' + data['adv_text']))
                if len(data['seq']) > self.max_len:
                    data['seq'] = data['seq'][:self.max_len]

    def __len__(self):
        if not self.raw:
            return len(self.data) // args.scale
        else:
            return len(self.data)

    def __getitem__(self, index):
        if not self.raw:
            return self.data[index * args.scale]
        else:
            return self.data[index]
# %%
class CarliniL2:

    def __init__(self, targeted=True, search_steps=None, max_steps=None, cuda=False, debug=False, num_classes=3):
        print("const confidence lr:", args.const, args.confidence, args.lr)
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
        self.seq = None
        self.seq_len = None
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
                    # print(self.wv[self.seq[0][j].item()])
                    # if self.seq[0][j].item() not in self.wv.keys():

                    similar_wv = model.bert.embeddings.word_embeddings(torch.LongTensor(self.wv[self.seq[i][j].item()]).cuda())
                    new_placeholder = input_adv[i, j].data
                    temp_place = new_placeholder.expand_as(similar_wv)
                    new_dist = torch.norm(temp_place - similar_wv.data, 2, -1)  # 2范数距离，一个字一个float
                    _, new_word = torch.min(new_dist, 0)
                    new_word_list.append(new_word.item())
                    # input_adv.data[j, i] = self.wv[new_word.item()].data
                    input_adv.data[i, j] = self.itereated_var.data[i, j] = similar_wv[new_word.item()].data
                    del temp_place
                batch_adv_sent.append(new_word_list)

            output = model(self.seq, self.seq_len, perturbed=input_adv)['pred']
            if args.debug_cw:
                print("output:", batch_adv_sent)
                print("input_adv:", input_adv)
                print("output:", output)
                adv_seq = torch.tensor(self.seq)
                for bi, (add_start, add_end) in enumerate(zip(self.batch_info['attack_start'], self.batch_info['attack_end'])):
                    adv_seq.data[bi, add_start:add_end] = torch.LongTensor(batch_adv_sent)
                print("out:", adv_seq)
                print("out embedding:", model.bert.embeddings.word_embeddings(adv_seq))
                out = model(adv_seq, self.seq_len)['pred']
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


def transform(seq, unk_words_dict=None, claim_seq_len=None):
    if claim_seq_len is None:
        claim_seq_len = len(seq)
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
            if x == 100 and idx < claim_seq_len and len(unk_words_dict[idx]) != 0:
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

# %%
def cw_word_attack(model, test_data):
    if args.untargeted:
        root_dir = os.path.join('./results', args.attack_model, args.function, 'untargeted')
    else:
        root_dir = os.path.join('./results', args.attack_model, args.function, 'targeted')
        
    print(f"Start attacking {args.model_name}, function={args.function}, {'untargeted' if args.untargeted else 'targeted'}, number of testing samples = {len(test_data)}")
    adv_correct = 0
    targeted_success = 0
    untargeted_success = 0
    orig_correct = 0
    tot = 0
    tot_diff = 0
    tot_len = 0
    tot_success = 0
    adv_pickle = []

    test_batch = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    cw = CarliniL2(debug=args.debugging, targeted=not args.untargeted, cuda=True, num_classes=5)
    for batch_index, batch in enumerate(tqdm(test_batch)):
        batch_add_start = batch['attack_start']
        batch_add_end = batch['attack_end']
        # for s, e in zip(batch['attack_start'], batch['attack_end']):
        #     batch['add_start'].append(1)
        #     batch['add_end'].append(l)

        data = batch['seq'] = torch.stack(batch['seq']).t().to(device)
        batch['claim_seq'] = torch.stack(batch['claim_seq']).t().to(device)
        orig_sent = transform(batch['seq'][0])

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
        out = model(batch['seq'], batch['seq_len'])['pred']
        prediction = torch.max(out, 1)[1]
        ori_prediction = prediction
        if ori_prediction[0].item() != label[0].item():
            continue
        batch['orig_correct'] = torch.sum((prediction == label).float())

        # prepare attack
        input_embedding = model.bert.embeddings.word_embeddings(data)
        cw_mask = np.zeros(input_embedding.shape).astype(np.float32)
        cw_mask = torch.from_numpy(cw_mask).float().to(device)
        for i, seq in enumerate(batch['seq_len']):
            cw_mask[i][1:seq] = 1

        if args.function == 'all':
            cluster_char_dict = get_similar_dict(batch['similar_dict'])
            bug_char_dict, unk_words_dict = get_bug_dict(batch['bug_dict'], batch['claim_seq'][0])
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
            all_dict, unk_words_dict = get_bug_dict(batch['bug_dict'], batch['claim_seq'][0])
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
        cw.seq = data
        cw.batch_info = batch
        cw.seq_len = seq_len

        # attack
        adv_data = cw.run(model, input_embedding, attack_targets)
        # retest
        adv_seq = torch.tensor(batch['seq']).to(device)
        for bi, (add_start, add_end) in enumerate(zip(batch_add_start, batch_add_end)):
            if bi in cw.o_best_sent:
                for i in range(add_start, add_end):
                    adv_seq.data[bi, i] = all_dict[adv_seq.data[bi, i].item()][cw.o_best_sent[bi][i - add_start]]

        out = model(adv_seq, seq_len)['pred']
        prediction = torch.max(out, 1)[1]
        orig_correct += batch['orig_correct'].item()
        adv_correct += torch.sum((prediction == label).float()).item()
        targeted_success += torch.sum((prediction == attack_targets).float()).item()
        untargeted_success += torch.sum((prediction != label).float()).item()
        tot += len(batch['label'])

        for i in range(len(batch['label'])):
            diff = difference(adv_seq[i], data[i])
            claim_seq_len = batch_add_end[i]
            adv_pickle.append({
                'index': batch_index,
                'adv_text': transform(adv_seq[i], unk_words_dict, claim_seq_len=claim_seq_len),
                'orig_text': transform(batch['seq'][i]),
                'raw_text': batch['text'][i],
                'label': label[i].item(),
                'target': attack_targets[i].item(),
                'ori_pred': ori_prediction[i].item(),
                'pred': prediction[i].item(),
                'diff': diff,
                'orig_seq': batch['seq'][i].cpu().numpy().tolist(),
                'adv_seq': adv_seq[i].cpu().numpy().tolist(),
                'seq_len': batch['seq_len'][i].item(),
                'claim_seq_len': claim_seq_len
            })
            # assert ori_prediction[i].item() == label[i].item()
            if (args.untargeted and prediction[i].item() != label[i].item()) or (not args.untargeted and prediction[i].item() == attack_targets[i].item()):
                tot_success += 1
                tot_diff += diff
                tot_len += claim_seq_len
                if batch_index % 100 == 0:
                    try:
                        print(("tot:", tot))
                        print(("avg_seq_len: {:.1f}".format(0 if tot_success == 0 else tot_len / tot_success)))
                        print(("avg_diff: {:.1f}".format(0 if tot_success == 0 else tot_diff / tot_success)))
                        print(("avg_diff_rate: {:.1f}%".format(0 if tot_len == 0 else tot_diff / tot_len * 100)))
                        print(("orig_correct: {:.1f}%".format(orig_correct / (batch_index + 1) * 100)))
                        print(("adv_correct: {:.1f}%".format(adv_correct / tot * 100)))
                        if args.untargeted:
                            print(("targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
                            print(("*untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
                        else:
                            print(("*targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
                            print(("untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
                    except:
                        continue
    joblib.dump(adv_pickle, os.path.join(args.root_dir, f"{'untargeted' if args.untargeted else 'targeted'}-{args.function}-{len(test_data)}-adv_text.pkl"))
    print(("tot:", tot))
    print(("avg_seq_len: {:.1f}".format(0 if tot_success == 0 else tot_len / tot_success)))
    print(("avg_diff: {:.1f}".format(0 if tot_success == 0 else tot_diff / tot_success)))
    print(("avg_diff_rate: {:.1f}%".format(0 if tot_len == 0 else tot_diff / tot_len * 100)))
    print(("orig_correct: {:.1f}%".format(orig_correct / len(test_data) * 100)))
    print(("adv_correct: {:.1f}%".format(adv_correct / tot * 100)))
    if args.untargeted:
        print(("targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
        print(("*untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
    else:
        print(("*targeted successful rate: {:.1f}%".format(targeted_success / tot * 100)))
        print(("untargeted successful rate: {:.1f}%".format(untargeted_success / tot * 100)))
    print(("const confidence:", args.const, args.confidence))

# %%
def check_model(model, test_data):
    model.eval()
    device = next(model.parameters()).device
    test_batch = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    correct = mistake = 0
    for batch_index, batch in enumerate(tqdm(test_batch)):
        batch['seq'] = torch.stack(batch['seq']).t().to(device)
        batch['seq_len'] = batch['seq_len'].to(device)
        label = batch['label'] = batch['label'].to(device)
        out = model(batch['seq'], batch['seq_len'])['pred']
        prediction = torch.max(out, 1)[1]
        ori_prediction = prediction
        if ori_prediction[0].item() != label[0].item():
            mistake += 1
        else:
            correct += 1
        batch['orig_correct'] = torch.sum((prediction == label).float())
    acc = correct/(correct + mistake)
    print(f"correct={correct}, mistake={mistake}, acc={acc}")

# %%
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
device = torch.device("cuda")

model = models.BertC(dropout=args.dropout, num_class=5)
try:
    model.load_state_dict(torch.load(args.load, map_location=device))
except RuntimeError:
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.load, map_location=device).items()})
model = model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
test_data = YelpDataset(args.test_data)

# %%
cw_word_attack(model, test_data)