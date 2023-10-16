import pandas as pd
import csv
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset, WeightedRandomSampler
# from transformers import RobertaTokenizer
# from transformers import BertTokenizer
from transformers import AutoTokenizer
from random import random,choice
import torch

class PretrainDataset(Dataset):
    def __init__(self, args, data_path, test_model = False):
        self.args = args
        # self.data = pd.read_csv(data_path,sep='\t', quoting=csv.QUOTE_NONE)
        # 读取路径里面所有的数据
        single_data = pd.concat([pd.read_csv(path,sep='\t', quoting=csv.QUOTE_NONE) for path in data_path])
        self.data = pd.concat([single_data for _ in range(args.copy_time)])
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_model
        self.stance2label = {"AGAINST":0,"FAVOR":1,"NONE":2}
        self.label2stance = {"0":"AGAINST","1":"FAVOR","2":"NONE"}
        self.premise = args.premise
        # self.premise2label = {"AGAINST":0,"FAVOR":1}
        self.claim_list = list(self.data.groupby('Claim').groups.keys())
        
    def __getitem__(self,index):
        text  = self.data['Tweet'].iloc[index]
        claim = self.data['Claim'].iloc[index]
        # MLM文本输入数据
        mlm_input_data = self.tokenizer(text,claim,max_length=self.bert_seq_length, \
                                    padding='max_length', \
                                    truncation='longest_first')
        mlm_input_ids = torch.tensor(mlm_input_data['input_ids'],dtype=torch.long)
        mlm_attention_mask = torch.tensor(mlm_input_data['attention_mask'],dtype=torch.long)
        
        # nsp_label = []
        if random() <self.args.negative_rate:
            claim_sample = claim
        else:
            claim_sample = choice(list(set(self.claim_list)-set([claim])))
        nsp_input_data = self.tokenizer(text,claim_sample,max_length=self.bert_seq_length, \
                                    padding='max_length', \
                                    truncation='longest_first')
        nsp_input_ids = torch.tensor(nsp_input_data['input_ids'],dtype=torch.long)
        nsp_attention_mask = torch.tensor(nsp_input_data['attention_mask'],dtype=torch.long)
        data = dict(
            nsp_input_ids = nsp_input_ids,
            nsp_attention_mask = nsp_attention_mask,
            mlm_input_ids = mlm_input_ids,
            mlm_attention_mask = mlm_attention_mask
        )
        # 加入Claim标识
        claim_id = self.claim_list.index(claim)
        claim_id = torch.tensor(claim_id,dtype=torch.long)
        data['claim'] = claim_id
        if self.test_mode:
            return data
        label = 1 if claim_sample==claim else 0
        label_id = torch.tensor(label,dtype=torch.long)
        data['nsp_label'] = label_id

        return data
    def __len__(self):
        return self.data.shape[0]
    
    @classmethod
    def create_dataloaders(cls, args):
        
        train_dataset = cls(args, args.train_path)
        args.copy_time = 1
        valid_dataset = cls(args, args.valid_path)
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
        
        train_dataloader = DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True,
                                        pin_memory=True)
        valid_dataloader = DataLoader(valid_dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=valid_sampler,
                                    drop_last=False,
                                    pin_memory=True)
        print('The train data length: ',len(train_dataloader))
        print('The valid data length: ',len(valid_dataloader))
        
        return train_dataloader, valid_dataloader

