import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig,RobertaModel
from transformers import BertConfig,BertModel
import numpy as np
from src.utils import FocalLoss
from transformers import AutoModel, AutoConfig
# from DeBERTa import deberta


class FtModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.pretrain_model_path is not None:
            pass
            # print(f"持续预训练模型路径:{args.pretrain_model_path}")
            # ckpoint = torch.load(args.pretrain_model_path)
            # self.roberta.load_state_dict(ckpoint["model_state_dict"])
        else:
            self.hidden_size = 768
            if "large" in  args.deberta_dir:
                self.hidden_size = 1024
            self.deberta = deberta.DeBERTa(pre_trained=args.deberta_dir)
            self.deberta.apply_state()
            # self.config = AutoConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
            # self.roberta = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache) 
        # self.att_head = AttentionHead(self.config.hidden_size * 4, self.config.hidden_size)
        if args.multi_task:
            self.stance_cls = nn.Linear(self.hidden_size, 3)
            self.premise_cls = nn.Linear(self.hidden_size, 2)
        else:
            if args.premise:
                args.class_label = 2
            self.cls = nn.Linear(self.hidden_size, args.class_label)
        # self.test_model = args.test_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if args.gamma_focal > 0:
            self.focal_loss = FocalLoss(class_num=args.class_label, gamma = args.gamma_focal)
        # self.NSP = args.NSP
        # if self.NSP:
        #     self.nsp_cls = nn.Linear(self.config.hidden_size, 2)
        
    def forward(self,input_data,inference=False):
        # input_ids = 
        hidden_states = self.deberta.bert(input_ids=input_data['input_ids'], attention_mask=input_data['attention_mask'])

        # pooler = outputs.pooler_output 
        # last_hidden_states = outputs.last_hidden_state
        # hidden_states = outputs.hidden_states
        h12 = hidden_states[-1]
        h11 = hidden_states[-2]
        h10 = hidden_states[-3]
        h09 = hidden_states[-4]
        
        # cat_hidd = torch.cat([h12,h11,h10,h09],dim=-1)
        # att_hidd = self.att_head(cat_hidd)
        
        h12_mean = torch.mean(h12 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h11_mean = torch.mean(h11 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h10_mean = torch.mean(h10 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h09_mean = torch.mean(h09 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        loss_clip = None
        # 对Claim做mean处理，获取Claim的特征
        if "roberta" in self.args.bert_dir:
            sep_index = (input_data['input_ids'] == 2).nonzero()
            sep_index = sep_index[:,1].view(-1,3) # bs * 3 一个sequence一定有三个<\s> 暂时对roberta有效
            claim_feat = torch.vstack([ torch.mean(h12[i,sep_index[i,1]+1:sep_index[i,2]],dim=0) for i in range(sep_index.size(0))]) # bs * hidden_size
            # 对Tweet text做mean处理，后去tweet的特征
            tweet_feat = torch.vstack([ torch.mean(h12[i,1:sep_index[i,0]],dim=0) for i in range(sep_index.size(0))]) # bs * hidden_size
        else:
            sep_index = (input_data['input_ids'] == 102).nonzero()
            sep_index = sep_index[:,1].view(-1,2) # bs * 2 一个sequence一定有三个[SEP] 暂时对bert有效
            claim_feat = torch.vstack([ torch.mean(h12[i,sep_index[i,0]+1:sep_index[i,1]],dim=0) for i in range(sep_index.size(0))]) # bs * hidden_size
                # 对Tweet text做mean处理，后去tweet的特征
            tweet_feat = torch.vstack([ torch.mean(h12[i,1:sep_index[i,0]],dim=0) for i in range(sep_index.size(0))]) # bs * hidden_size
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
        
        
        # cat_output = torch.cat([h12_mean,h11_mean,h10_mean,h09_mean,att_hidd],dim = -1)
        # cat_output = torch.cat([h12_mean,att_hidd],dim = -1)
        if self.args.multi_task:
            # stance
            stance_logits = self.stance_cls(h12_mean)
            stance_probability = nn.functional.softmax(stance_logits)
            loss_s, logits_s, accuracy_s, pred_label_id_s = self.cal_loss(stance_logits, input_data['label'])
            # premise
            premise_logits = self.premise_cls(h12_mean)
            premise_probability = nn.functional.softmax(premise_logits)
            loss_p, logits_p, accuracy_p, pred_label_id_p = self.cal_loss(premise_logits, input_data['premise_label'])
            if inference:
                return premise_probability  if self.args.premise else stance_probability
            if self.args.premise:
                loss, logits, accuracy, pred_label_id = loss_p+loss_s, logits_p, accuracy_p, pred_label_id_p
            else:
                loss, logits, accuracy, pred_label_id = loss_p+loss_s, logits_s, accuracy_s, pred_label_id_s
            
        else:
            logits = self.cls(h12_mean)
            probability = nn.functional.softmax(logits)
            if inference:
                return probability
            if self.args.premise:
                loss, logits, accuracy, pred_label_id = self.cal_loss(logits, input_data['premise_label'])
            else:
                loss, logits, accuracy, pred_label_id = self.cal_loss(logits, input_data['label'])
        if loss_clip is not None:
            loss += loss_clip*0.0

        return loss, logits, accuracy, pred_label_id
        
        
    # @staticmethod
    def cal_loss(self, logits, label):
        # label = label.squeeze(dim=1)
        if self.args.gamma_focal > 0:
            loss = self.focal_loss(logits, label)
        else:
            loss = F.cross_entropy(logits, label)
        with torch.no_grad():
            pred_label_id = torch.argmax(logits, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]

        return loss, logits, accuracy, pred_label_id
    
    
class AttentionHead(nn.Module):
    def __init__(self, cat_size, hidden_size=768):
        super().__init__()
        self.W = nn.Linear(cat_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)        
        
    def forward(self, hidden_states):
        att = torch.tanh(self.W(hidden_states))
        score = self.V(att)
        att_w = torch.softmax(score, dim=1)
        context_vec = att_w * hidden_states
        context_vec = torch.sum(context_vec,dim=1)
        
        return context_vec
    
    
# # NSP
# if self.NSP:
#     outputs_nsp = self.roberta(input_data['nsp_input_ids'], input_data['nsp_attention_mask'], output_hidden_states = True)
#     # nsp_pooler = outputs_nsp.pooler_output 
#     nsp_h12 = outputs.last_hidden_state
#     nsp_h12_mean = torch.mean(nsp_h12 * input_data['nsp_attention_mask'].unsqueeze(-1) , dim=1)
#     nsp_logits = self.nsp_cls(nsp_h12_mean)
#     nsp_loss = F.cross_entropy(nsp_logits, input_data['nsp_label'])
# if self.NSP:
#     loss += nsp_loss   