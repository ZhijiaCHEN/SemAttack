import numpy as np
from torch import nn
from torch.nn import BatchNorm1d, Linear, ReLU
from torch.autograd import Variable
from pytorch_transformers.modeling_bert import *
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer

def seq_len_to_mask(seq_len, max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, "seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        if max_len is None:
            max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)

    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, "seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        if max_len is None:
            max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

class BertEmbeddings_attack(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings_attack, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None, perturbed=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        if perturbed is not None:
            # print("output embedding:", perturbed)
            embeddings = perturbed + position_embeddings + token_type_embeddings
        else:
            # print("out embedding:", words_embeddings)
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModel_attack(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel_attack, self).__init__(config)

        self.embeddings = BertEmbeddings_attack(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None,
                perturbed=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                           perturbed=perturbed)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertC(nn.Module):
    def __init__(self, name='bert-base-uncased', dropout=0.1, num_class=5):
        super(BertC, self).__init__()
        config = BertConfig.from_pretrained(name, num_labels=num_class)
        self.bert = BertModel_attack(config)
        self.proj = nn.Linear(config.hidden_size, num_class)
        self.loss_f = nn.CrossEntropyLoss()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, src, seq_len, gold=None, perturbed=None):
        src_mask = seq_len_to_mask(seq_len, src.size(1))
        out = self.bert(src, attention_mask=src_mask, perturbed=perturbed)
        embed = out[1]
        # print(embed.size())
        logits = self.proj(self.drop(embed))
        ret = {"pred": logits}
        if gold is not None:
            ret["loss"] = self.loss_f(logits, gold)
        ret['embedding'] = out[0]
        return ret

def kernal_mus(n_kernels):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu

def kernel_sigmas(n_kernels):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

class inference_model(nn.Module):
    def __init__(self, bert_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = bert_model
        self.evi_num = args.evi_num
        self.nlayer = args.layer
        self.kernel = args.kernel
        self.proj_inference_de = nn.Linear(self.bert_hidden_dim * 2, self.num_labels)
        self.proj_att = nn.Linear(self.kernel, 1)
        self.proj_input_de = nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim)
        self.proj_gat = nn.Sequential(
            Linear(self.bert_hidden_dim * 2, 128),
            ReLU(True),
            Linear(128, 1)
        )
        self.proj_select = nn.Linear(self.kernel, 1)
        self.mu = Variable(torch.FloatTensor(kernal_mus(self.kernel)), requires_grad = False).view(1, 1, 1, 21).cuda()
        self.sigma = Variable(torch.FloatTensor(kernel_sigmas(self.kernel)), requires_grad = False).view(1, 1, 1, 21).cuda()


    def self_attention(self, inputs, inputs_hiddens, mask, mask_evidence, index):
        idx = torch.LongTensor([index]).cuda()
        mask = mask.view([-1, self.evi_num, self.max_len])
        mask_evidence = mask_evidence.view([-1, self.evi_num, self.max_len])
        own_hidden = torch.index_select(inputs_hiddens, 1, idx)
        own_mask = torch.index_select(mask, 1, idx)
        own_input = torch.index_select(inputs, 1, idx)
        own_hidden = own_hidden.repeat(1, self.evi_num, 1, 1)
        own_mask = own_mask.repeat(1, self.evi_num, 1)
        own_input = own_input.repeat(1, self.evi_num, 1)

        hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=-1)
        own_norm = F.normalize(own_hidden, p=2, dim=-1)

        att_score = self.get_intersect_matrix_att(hiddens_norm.view(-1, self.max_len, self.bert_hidden_dim), own_norm.view(-1, self.max_len, self.bert_hidden_dim),
                                                  mask_evidence.view(-1, self.max_len), own_mask.view(-1, self.max_len))
        att_score = att_score.view(-1, self.evi_num, self.max_len, 1)
        #if index == 1:
        #    for i in range(self.evi_num):
        #print (att_score.view(-1, self.evi_num, self.max_len)[0, 1, :])
        denoise_inputs = torch.sum(att_score * inputs_hiddens, 2)
        weight_inp = torch.cat([own_input, inputs], -1)
        weight_inp = self.proj_gat(weight_inp)
        weight_inp = F.softmax(weight_inp, dim=1)
        outputs = (inputs * weight_inp).sum(dim=1)
        weight_de = torch.cat([own_input, denoise_inputs], -1)
        weight_de = self.proj_gat(weight_de)
        weight_de = F.softmax(weight_de, dim=1)
        outputs_de = (denoise_inputs * weight_de).sum(dim=1)
        return outputs, outputs_de

    def get_intersect_matrix(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1], 1)
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu.cuda()) ** 2) / (self.sigma.cuda() ** 2) / 2)) * attn_d
        pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(pooling_sum, min=1e-10)) * attn_q
        log_pooling_sum = torch.sum(log_pooling_sum, 1) / (torch.sum(attn_q, 1) + 1e-10)
        log_pooling_sum = self.proj_select(log_pooling_sum).view([-1, 1])
        return log_pooling_sum

    def get_intersect_matrix_att(self, q_embed, d_embed, attn_q, attn_d):
        attn_q = attn_q.view(attn_q.size()[0], attn_q.size()[1])
        attn_d = attn_d.view(attn_d.size()[0], 1, attn_d.size()[1], 1)
        sim = torch.bmm(q_embed, torch.transpose(d_embed, 1, 2)).view(q_embed.size()[0], q_embed.size()[1], d_embed.size()[1], 1)
        pooling_value = torch.exp((- ((sim - self.mu.cuda()) ** 2) / (self.sigma.cuda() ** 2) / 2)) * attn_d
        log_pooling_sum = torch.sum(pooling_value, 2)
        log_pooling_sum = torch.log(torch.clamp(log_pooling_sum, min=1e-10))
        log_pooling_sum = self.proj_att(log_pooling_sum).squeeze(-1)
        log_pooling_sum = log_pooling_sum.masked_fill_((1 - attn_q).bool(), -1e4)
        log_pooling_sum = F.softmax(log_pooling_sum, dim=1)
        return log_pooling_sum

    def forward(self, inputs):
        inp_tensor, msk_tensor, seg_tensor = inputs
        msk_tensor = msk_tensor.view(-1, self.max_len)
        inp_tensor = inp_tensor.view(-1, self.max_len)
        seg_tensor = seg_tensor.view(-1, self.max_len)
        inputs_hiddens, inputs = self.pred_model(inp_tensor, msk_tensor, seg_tensor)
        mask_text = msk_tensor.view(-1, self.max_len).float()
        mask_text[:, 0] = 0.0
        mask_claim = (1 - seg_tensor.float()) * mask_text
        mask_evidence = seg_tensor.float() * mask_text
        inputs_hiddens = inputs_hiddens.view(-1, self.max_len, self.bert_hidden_dim)
        inputs_hiddens_norm = F.normalize(inputs_hiddens, p=2, dim=2)
        log_pooling_sum = self.get_intersect_matrix(inputs_hiddens_norm, inputs_hiddens_norm, mask_claim, mask_evidence)
        log_pooling_sum = log_pooling_sum.view([-1, self.evi_num, 1])
        select_prob = F.softmax(log_pooling_sum, dim=1)
        inputs = inputs.view([-1, self.evi_num, self.bert_hidden_dim])
        inputs_hiddens = inputs_hiddens.view([-1, self.evi_num, self.max_len, self.bert_hidden_dim])
        inputs_att_de = []
        for i in range(self.evi_num):
            outputs, outputs_de = self.self_attention(inputs, inputs_hiddens, mask_text, mask_text, i)
            inputs_att_de.append(outputs_de)
        inputs_att = inputs.view([-1, self.evi_num, self.bert_hidden_dim])
        inputs_att_de = torch.cat(inputs_att_de, dim=1)
        inputs_att_de = inputs_att_de.view([-1, self.evi_num, self.bert_hidden_dim])
        inputs_att = torch.cat([inputs_att, inputs_att_de], -1)
        inference_feature = self.proj_inference_de(inputs_att)
        class_prob = F.softmax(inference_feature, dim=2)
        prob = torch.sum(select_prob * class_prob, 1)
        prob = torch.log(prob)
        return prob
    
    def word_embeddings(self, input_ids):
        return self.pred_model(input_ids, None, None)

class Pretrained_Fever_BERT(nn.Module):
    def __init__(self, states):
        super(Pretrained_Fever_BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=3)
        self.model.load_state_dict(states)
        self.loss_f = nn.CrossEntropyLoss()
    
    def forward(self, src, seq_len, gold=None, perturbed=None):
        assert perturbed is None
        src_mask = seq_len_to_mask(seq_len, src.size(1))
        out = self.model(src, attention_mask=src_mask, labels=gold, output_hidden_states=True)
        # print(embed.size())
        ret = {"pred": out.logits}
        ret["loss"] = out.loss
        ret['embedding'] = out.hidden_states[-1]
        return ret

    def word_embeddings(self, input_ids):
        return self.model.bert(input_ids, None, None)
    
    # def eval(self):
    #     self.bert.eval()
    
    # def to(self, device):
    #     self.bert = self.bert.to(device)