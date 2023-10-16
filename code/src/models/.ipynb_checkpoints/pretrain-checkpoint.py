import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (RobertaTokenizer,
                          RobertaConfig,
                          RobertaModel,
                          RobertaForMaskedLM)
from transformers.models.roberta.modeling_roberta import RobertaPooler
from transformers.models.bert.modeling_bert import BertPooler

import numpy as np
from src.dataset import MaskLM
from transformers import AutoModel, AutoConfig
from transformers import (BertTokenizer,
                          BertConfig,
                          BertModel,
                          BertForMaskedLM)

class PretrainRoberta_sep(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        if "roberta" in  args.bert_dir:
            self.mlm  = RobertaForMaskedLM.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        else:
            self.mlm  = BertForMaskedLM.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.NSP = self.args.NSP
        if args.NSP:
            if "roberta" in  args.bert_dir:
                self.pooler = RobertaPooler(self.config)
            else:
                self.pooler = BertPooler(self.config)
                
            self.cls = nn.Linear(self.config.hidden_size, 2)
        self.lm = MaskLM(tokenizer_path=args.bert_dir)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_data):
#         input_ids, lm_label = [], []
#         for i in range(self.args.copy_time):
#             temp_input_ids, temp_lm_label = self.lm.torch_mask_tokens(input_data["input_ids"].cpu())
#             input_ids.append(temp_input_ids)
#             lm_label.append(temp_lm_label)
        
#         input_ids = torch.vstack(input_ids)
#         lm_label = torch.vstack(lm_label)
        # MLM
        input_ids, lm_label = self.lm.torch_mask_tokens(input_data["mlm_input_ids"].cpu())
        mlm_input_ids = input_ids.to(input_data["mlm_input_ids"].device)
        lm_label = lm_label.to(input_data["mlm_input_ids"].device) 
        
        mlm_outputs = self.mlm(input_ids=mlm_input_ids, \
                           attention_mask=input_data["mlm_attention_mask"], \
                           labels = lm_label,\
                           output_hidden_states=True)
        mlm_loss = mlm_outputs.loss
        # hidden_states = outputs.hidden_states
        # sequence_output  = outputs.hidden_states[-1]
        loss_clip = None
        if False:
            # 对Claim做mean处理，获取Claim的特征
            sep_index = (input_data['input_ids'] == 2).nonzero()
            sep_index = sep_index[:,1].view(-1,3) # bs * 3 一个sequence一定有三个<\s> 暂时对roberta有效
            claim_feat = torch.vstack([ torch.mean(sequence_output[i,sep_index[i,1]+1:sep_index[i,2]],dim=0) for i in range(sep_index.size(0))]) # bs * hidden_size
            # 对Tweet text做mean处理，后去tweet的特征
            tweet_feat = torch.vstack([ torch.mean(sequence_output[i,1:sep_index[i,0]],dim=0) for i in range(sep_index.size(0))]) # bs * hidden_size


            # normalized features
            claim_feat = claim_feat / claim_feat.norm(dim=1, keepdim=True)
            tweet_feat = tweet_feat / tweet_feat.norm(dim=1, keepdim=True)
            # cosine similarity as logits
            logit_scale = self.logit_scale.exp()
            logits_per_claim = logit_scale * claim_feat @ tweet_feat.t()
            logits_per_tweet = logits_per_claim.t()

            # calculate loss
            # logits_matrix = torch.matmul(features_T, features_V.t())
            # labels = torch.arange(0,logits_per_tweet.size(0)).long().cuda()
            labels = torch.zeros((logits_per_tweet.size(0),logits_per_tweet.size(0)),dtype=torch.float).cuda()
            for i in range(logits_per_tweet.size(0)):
                for j in range(i,logits_per_tweet.size(0)):
                    labels[i][j] = 1 if input_data["claim"][i]==input_data["claim"][j] else 0
            loss_fc = nn.CrossEntropyLoss()
            loss_T = loss_fc(logits_per_tweet, labels)
            loss_C = loss_fc(logits_per_claim, labels)
            loss_clip = (loss_T+loss_C)/2
            
            mlm_loss += loss_clip*0.2
        
        
        # 
        nsp_loss = None
        if self.args.NSP:
            nsp_outputs = self.mlm(input_ids=input_data["nsp_input_ids"], \
                           attention_mask=input_data["nsp_attention_mask"], \
                           labels = lm_label,\
                           output_hidden_states=True)
            sequence_output = nsp_outputs.hidden_states[-1]
            pooler_out = self.pooler(sequence_output)
            nsp_logits = self.cls(pooler_out)
            nsp_loss = F.cross_entropy(nsp_logits, input_data["nsp_label"])
        if nsp_loss is not None:
            loss_dict = dict(
                mlm_loss = mlm_loss,
                nsp_loss = nsp_loss,
                loss = mlm_loss + nsp_loss
            )
            return loss_dict
        else:
            loss_dict = dict(
                loss = mlm_loss,
                mlm_loss = mlm_loss,
                nsp_loss = mlm_loss
            )
            return loss_dict
        
        
class PretrainRoberta(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        if "roberta" in  args.bert_dir:
            self.mlm  = RobertaForMaskedLM.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        else:
            self.mlm  = BertForMaskedLM.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.NSP = self.args.NSP
        if args.NSP:
            if "roberta" in  args.bert_dir:
                self.pooler = RobertaPooler(self.config)
            else:
                self.pooler = BertPooler(self.config)
                
            self.cls = nn.Linear(self.config.hidden_size, 2)
        self.lm = MaskLM(tokenizer_path=args.bert_dir)

    def forward(self, input_data):
        # MLM
        input_ids, lm_label = self.lm.torch_mask_tokens(input_data["nsp_input_ids"].cpu())
        mlm_input_ids = input_ids.to(input_data["nsp_input_ids"].device)
        lm_label = lm_label.to(input_data["nsp_input_ids"].device) 
        
        mlm_outputs = self.mlm(input_ids=mlm_input_ids, \
                           attention_mask=input_data["nsp_attention_mask"], \
                           labels = lm_label,\
                           output_hidden_states=True)
        mlm_loss = mlm_outputs.loss
        # hidden_states = outputs.hidden_states
        # sequence_output  = outputs.hidden_states[-1]
        loss_clip = None
        # 
        nsp_loss = None
        if self.args.NSP:
            # nsp_outputs = self.mlm(input_ids=input_data["nsp_input_ids"], \
            #                attention_mask=input_data["nsp_attention_mask"], \
            #                labels = lm_label,\
            #                output_hidden_states=True)
            sequence_output = mlm_outputs.hidden_states[-1]
            pooler_out = self.pooler(sequence_output)
            nsp_logits = self.cls(pooler_out)
            nsp_loss = F.cross_entropy(nsp_logits, input_data["nsp_label"])
        if nsp_loss is not None:
            loss_dict = dict(
                mlm_loss = mlm_loss,
                nsp_loss = nsp_loss,
                loss = mlm_loss + nsp_loss
            )
            return loss_dict
        else:
            loss_dict = dict(
                loss = mlm_loss,
                mlm_loss = mlm_loss,
                nsp_loss = mlm_loss
            )
            return loss_dict