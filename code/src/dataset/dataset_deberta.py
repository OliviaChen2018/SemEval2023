import pandas as pd
import csv
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset, WeightedRandomSampler
# from transformers import RobertaTokenizer
# from transformers import BertTokenizer
# from transformers import AutoTokenizer
from random import random,choice
from sklearn.model_selection import StratifiedKFold
import torch
from DeBERTa import deberta


class FinetuneDataset(Dataset):
    def __init__(self, args, data_path, test_model = False):
        self.args = args
        self.data = pd.read_csv(data_path,sep='\t', quoting=csv.QUOTE_NONE)
        # self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        # 
        self.vocab_path, self.vocab_type = deberta.load_vocab(pretrained_id=args.deberta_dir)
        self.tokenizer = deberta.tokenizers[self.vocab_type](self.vocab_path)
        
        
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_model
        self.stance2label = {"AGAINST":0,"FAVOR":1,"NONE":2}
        self.label2stance = {"0":"AGAINST","1":"FAVOR","2":"NONE"}
        self.premise = args.premise
        # self.premise2label = {"AGAINST":0,"FAVOR":1}
        self.claim_list = list(self.data.groupby('Claim').groups.keys())
        self.NSP = args.NSP
        
    def __getitem__(self,index):
        text  = self.data['Tweet'].iloc[index]
        claim = self.data['Claim'].iloc[index]
        # tokenize
        text_tokens = self.tokenizer.tokenize(text)
        claim_tokens = self.tokenizer.tokenize(claim)
        # truncation
        text_tokens = text_tokens[0:self.bert_seq_length-3-len(claim_tokens)]
        # concat
        tokens = "[CLS]" + text_tokens + "[SEP]" + claim_tokens + "[SEP]"
        
        # input_data = self.tokenizer(text,claim,max_length=self.bert_seq_length, \
        #                             padding='max_length', \
        #                             truncation='longest_first')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0]*(len(text_tokens)+2) + [1] *(len(claim_tokens)+1)
        attention_ids = [1] * len(tokens)
        assert len(input_ids) == len(segment_ids) == len(attention_ids)
        
        # padding
        padding_length = self.bert_seq_length - len(input_ids)
        input_ids += [0] * padding_length
        segment_ids += [0] * padding_length
        attention_ids += [0] * padding_length
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_ids, dtype=torch.long)
        
        data = dict(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        # 构建NSP任务数据
        if self.NSP:
            if random() < 0.5:
                claim_sample = claim
            else:
                claim_sample = choice(list(set(self.claim_list)-set([claim])))
            nsp_input_data = self.tokenizer(text,claim_sample,max_length=self.bert_seq_length, \
                                        padding='max_length', \
                                        truncation='longest_first')
            data["nsp_input_ids"] = torch.tensor(nsp_input_data['input_ids'],dtype=torch.long)
            data["nsp_attention_mask"] = torch.tensor(nsp_input_data['attention_mask'],dtype=torch.long)
            label_nsp = 1 if claim_sample==claim else 0
            label_nsp_id = torch.tensor(label_nsp,dtype=torch.long)
            data['nsp_label'] = label_nsp_id
        
        # 加入Claim标识
        claim_id = self.claim_list.index(claim)
        claim_id = torch.tensor(claim_id,dtype=torch.long)
        data['claim'] = claim_id
        if self.test_mode:
            return data
        # task2a和task2b 多任务训练
        premise = self.data['Premise'].iloc[index]
        premise_label_id = torch.tensor(premise,dtype=torch.long)
        data['premise_label'] = premise_label_id
            
        stance = self.data['Stance'].iloc[index]
        label = self.stance2label[stance]
        label_id = torch.tensor(label,dtype=torch.long)
        data['label'] = label_id
        
        
        return data
    def __len__(self):
        return self.data.shape[0]

    @classmethod
    def create_k_fold_dataloaders(cls, args):
        if args.k_fold > 1:
            print(f"进行{args.k_fold}折训练！")
            dataset = cls(args, args.train_path)
            skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
            # label = []
            # print()
            # for i in range(len(dataset)):
            #     print(dataset.data["Claim"].iloc[i])
            #     print(dataset.stance2label[dataset.data["Claim"].iloc[i]])
            #     label.append(dataset.stance2label[dataset.data["Claim"].iloc[i]])
            
            label = [dataset.data["Premise"].iloc[i] if args.premise else dataset.stance2label[dataset.data["Stance"].iloc[i]] for i in range(len(dataset))]
            for train_idx, valid_idx in skf.split(dataset.data, label):
                train_dataset = Subset(dataset, train_idx)
                valid_dataset = Subset(dataset, valid_idx)

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
                yield train_dataloader, valid_dataloader
        else:
            train_dataset = cls(args, args.train_path)
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
    
    @classmethod
    def create_dataloaders(cls, args):
        train_dataset = cls(args, args.train_path)
        valid_dataset = cls(args, args.valid_path)
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)
        
        train_dataloader = DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=False,
                                        pin_memory=True)
        valid_dataloader = DataLoader(valid_dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=valid_sampler,
                                    drop_last=False,
                                    pin_memory=True)
        print('The train data length: ',len(train_dataloader))
        print('The valid data length: ',len(valid_dataloader))
        
        return train_dataloader, valid_dataloader
