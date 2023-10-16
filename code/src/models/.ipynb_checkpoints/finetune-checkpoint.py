import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaConfig,RobertaModel
from transformers import BertConfig,BertModel
import numpy as np
from src.utils import FocalLoss
from transformers import AutoModel, AutoConfig

class FtModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if args.pretrain_model_path is not None:
            print(f"持续预训练模型路径:{args.pretrain_model_path}")
            if "roberta" in  args.bert_dir:
                self.config = RobertaConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
                self.roberta = RobertaModel(self.config, add_pooling_layer=False) 
            else:
                self.config = BertConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
                self.roberta = BertModel(self.config, add_pooling_layer=False) 
            
            ckpoint = torch.load(args.pretrain_model_path)
            self.roberta.load_state_dict(ckpoint["model_state_dict"])
        else:
            self.config = AutoConfig.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
            self.roberta = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache) 
            
        # self.att_head = AttentionHead(self.config.hidden_size * 4, self.config.hidden_size)
        self.cls = nn.Linear(self.config.hidden_size, args.class_label)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if args.gamma_focal > 0:
            self.focal_loss = FocalLoss(class_num=args.class_label, gamma = args.gamma_focal)     
    def forward(self,input_data,inference=False):
        # input_ids = 
        outputs = self.roberta(input_data['input_ids'], input_data['attention_mask'], output_hidden_states = True)

        pooler = outputs.pooler_output 
        last_hidden_states = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        h12 = hidden_states[-1]
        h11 = hidden_states[-2]
        h10 = hidden_states[-3]
        h09 = hidden_states[-4]
        # cls_concat = torch.cat([h12[:,0],h11[:,0],h10[:,0],h09[:,0]],1)
        # cat_hidd = torch.cat([h12,h11,h10,h09],dim=-1)
        # att_hidd = self.att_head(cat_hidd)
        
        h12_mean = torch.mean(h12 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h11_mean = torch.mean(h11 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h10_mean = torch.mean(h10 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        h09_mean = torch.mean(h09 * input_data['attention_mask'].unsqueeze(-1) , dim=1)
        loss_clip = None
        if self.args.multi_task:
            pass
        else:
            # h12_mean = torch.mean(h12,dim=1)
            # concat_out = torch.cat([pooler,att_hidd],dim=-1)
            concat_out = pooler
            # print(concat_out.size())
            logits = self.cls(concat_out)
            # probability = nn.functional.sigmoid(logits)
            # A = torch.tensor(4.0,dtype=torch.long)
            # bias = torch.tensor(1.0,dtype=torch.long)
            # score = probability * 4.0 + 1.0
            score = logits
            if inference:
                return score
            loss, score = self.cal_loss(score, input_data['label'])
            
        # if loss_clip is not None:
        #     loss += loss_clip*0.3
        return loss, score
        
        
    # @staticmethod
    def cal_loss(self, logits, label):
        # label = label.squeeze(dim=1)
        if self.args.gamma_focal > 0:
            loss = self.focal_loss(logits, label)
        else:
            loss_fc = torch.nn.MSELoss()
            logits = logits.to(torch.float)
            # print(logits)
            # print(label)
            loss = loss_fc(logits, label)
            # print(loss)
        return loss, logits
    

    
class FocalMSE:
    def __init__(self,gamma = 1.0):
        self.gamma = gamma
    def forward(self,score,label):
        pass
        return 
    
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