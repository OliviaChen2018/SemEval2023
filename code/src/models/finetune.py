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
            
#            让dropout概率为0
#             self.config.hidden_dropout_prob = 0
#             self.config.attention_probs_dropout_prob = 0
            
            self.roberta = AutoModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache) 
            
        # self.att_head = AttentionHead(self.config.hidden_size * 4, self.config.hidden_size)
        self.cls = nn.Linear(self.config.hidden_size, args.class_label)
#         self.cls_mid =  nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.ln = nn.LayerNorm(self.config.hidden_size)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # temperature
        if args.gamma_focal > 0:
            self.focal_loss = FocalLoss(class_num=args.class_label, gamma = args.gamma_focal)     
    def forward(self,input_data,inference=False):  #input_data从哪里传进来？通过model(batch)
        outputs = self.roberta(input_data['input_ids'], input_data['attention_mask'], output_hidden_states = True)

        pooler = outputs.pooler_output 
        last_hidden_states = outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        h12 = hidden_states[-1]  # 由于共有13个隐藏层，最后1个隐藏层就是第13个隐藏层，index12
        # h12其实就是 last_hidden_states？
        h11 = hidden_states[-2]  # 倒数第2个隐藏层的index11
        h10 = hidden_states[-3]  # 倒数第3个隐藏层的index10
        h09 = hidden_states[-4]  # 倒数第4个隐藏层的index9
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
#             concat_out=self.ln(concat_out)
            #concat_out=self.cls_mid(concat_out)
            logits = self.cls(concat_out) # 这是一个全连接层，得到包含概率的logits结果
            # probability = nn.functional.sigmoid(logits)
            # A = torch.tensor(4.0,dtype=torch.long)
            # bias = torch.tensor(1.0,dtype=torch.long)
            # score = probability * 4.0 + 1.0
            
#             logits = torch.where(torch.isinf(logits), torch.full_like(logits, 0), logits)
            score = logits
            if inference:
                return score
            loss, score = self.cal_loss(score, input_data['label'])
#             loss, mse, score = self.cal_loss(score, input_data['label'])
            
        # if loss_clip is not None:
        #     loss += loss_clip*0.3
        return loss, score
#         return loss, mse, score
        
        
    # @staticmethod
    def cal_loss(self, logits, label):
        # label = label.squeeze(dim=1)
        if self.args.gamma_focal > 0:
            loss = self.focal_loss(logits, label)
        else:
#             loss_fc = torch.nn.MSELoss()
#             loss_fc = FocalMSE()  # 初始化一个实例对象
            loss_fc = FocalMSE(gamma=5.0)  # 初始化一个实例对象
#             loss_fc = ContrastiveLoss()
            logits = logits.to(torch.float)
#             print(logits)
#             print(label)
            loss = loss_fc(logits, label)   # 调用forward函数
#             loss, logits = loss_fc(logits, label)   # 调用forward函数
#             loss, mse, logits = loss_fc(logits, label)   # 调用forward函数
            # print(loss)
        return loss, logits
#             return loss, mse, logits
    

    
# class FocalMSE:
#     def __init__(self,gamma = 1.0):
#         self.gamma = gamma
#     def forward(self,score,label):
#         pass
#         return 
class FocalMSE(nn.Module):
    def __init__(self, gamma = 1.0):
        super(FocalMSE, self).__init__()
        self.gamma = gamma
    def forward(self, logits, label):
        # loss_fc = torch.nn.MSEloss()
        logits = logits.to(torch.float)
        
        B = logits.size(0) #返回logits第0维的数量，即logits中的行数。此处表示样本个数。
        logits = logits.view(-1)
        # logits = torch.clamp(logits, max = 5)
        label = label.view(-1)
        #focal_factor = torch.pow(torch.abs(logits - label), self.gamma)
        #focal_factor = torch.exp(torch.abs(logits - label))
        sim = torch.cosine_similarity(logits, label, dim=0)
        
        if sim.item()<0:
            f = -sim
#             print("sim小于0了")
        else:
            f = torch.tensor(1)
#             f = torch.exp(sim)
#             focal_factor = torch.exp(sim)
        focal_factor = torch.pow(f, self.gamma)
        
#         focal_factor = torch.exp(torch.pow(f, self.gamma)) #logits - label反应的是困难样本
        # L1_difference = torch.abs(logits - label)
        # L1_probability = L1_difference/L1_difference.sum()
        # L1_probability = F.softmax(L1_difference)
        # focal_factor = torch.tan((torch.pi/2) * L1_probability)
        
        mse_loss = torch.div(torch.pow(torch.abs(logits - label), 2), B) #均方差的公式
        focal_mse = torch.mul(focal_factor, mse_loss).sum() #在均方差的基础上乘一个系数
        return focal_mse, logits
#         return focal_mse, mse, logits
    
    
# class ContrastiveLoss(nn.Module):
#     def __init__(self, gamma = 1.0, tao=1.0):
#         super(ContrastiveLoss, self).__init__()
#         self.gamma = gamma
#         self.tao = tao
#     def forward(self, logits, label):
#         # loss_fc = torch.nn.MSEloss()
#         logits = logits.to(torch.float)
#         logits = logits.view(-1)
#         label = label.view(-1)
#         sim = torch.cosine_similarity(logits, label, dim=0)
        
#         e_sim = torch.exp(torch.div(sim, self.tao))
#         sim_sum = e_sim.sum()
#         sim_div = torch.div(e_sim, sim_sum)
# #         con_loss = torch.log(sim_div)*(-1)
#         con_loss = sim_div*(-1)
#         return con_loss, logits

class ContrastiveLoss(nn.Module):
    def __init__(self, device='cuda', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.T = temperature  # 超参数 温度
#         self.register_buffer("temperature", torch.tensor(temperature).to(device))		
    def forward(self, logits, label):
        n = label.shape[0]
        #这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(logits.unsqueeze(1), logits.unsqueeze(0), dim=2).cuda()
        #这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask
        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = (torch.ones(n ,n) - torch.eye(n, n)).cuda()
        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/self.T)
        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
#         similarity_matrix = similarity_matrix.cuda()
#         mask_dui_jiao_0 = mask_dui_jiao_0.cuda()
        similarity_matrix = similarity_matrix*mask_dui_jiao_0
        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix
        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim
        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1)
        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum)
        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        mask_no_sim = mask_no_sim.cuda()
        loss = loss.cuda()
        
        loss = mask_no_sim + loss + torch.eye(n, n).cuda()
        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  #求-log
        loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n
        return loss
        
    
# class ContrastiveLoss(nn.Module):
#     def __init__(self, batch_size, device='cuda', temperature=0.5):
#         super().__init__()
#         self.batch_size = batch_size
#         self.temperature = temperature  # 超参数 温度
# #         self.register_buffer("temperature", torch.tensor(temperature).to(device))			
#         self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		# 主对角线为0，其余位置全为1的mask矩阵
        
#     def forward(self, emb_i, emb_j):		# emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
#         z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
#         z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)

#         representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
#         similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        
#         sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
#         sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
#         positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        
#         nominator = torch.exp(positives / self.temperature)             # 2*bs
#         denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
    
#         loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
#         loss = torch.sum(loss_partial) / (2 * self.batch_size)
#         return loss
    
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