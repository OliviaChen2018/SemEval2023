import pandas as pd
import csv
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset, WeightedRandomSampler, ConcatDataset
# from transformers import RobertaTokenizer
# from transformers import BertTokenizer
from transformers import AutoTokenizer
from random import random,choice,sample
from sklearn.model_selection import StratifiedKFold
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



class FinetuneDataset(Dataset):
    def __init__(self, args, data_path=None, test_model = False, data = None):
        self.args = args
        if data is not None:
            self.data = data
        elif data_path is not None:
            self.data = pd.read_csv(data_path,sep=',')
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)  # 初始化一个tokenizer
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_model
        
    def __getitem__(self,index):
    # 使用dataset[i]的时候会调用__getitem__函数，然后会自动调用getBertData获取index对应的data。
    # data包括两部分：input_ids和attention_mask。
        text  = self.data['text'].iloc[index]
        return self.getBertData(index,text)
        
    
    def __len__(self):
        return self.data.shape[0]
    
    def getBertData(self,index, text):  #这个index参数其实没有用
        input_data = self.tokenizer(text,max_length=self.bert_seq_length, \
                                    padding='max_length', \
                                    truncation='longest_first')
        
        input_ids = torch.tensor(input_data['input_ids'],dtype=torch.long)
        attention_mask = torch.tensor(input_data['attention_mask'],dtype=torch.long)
        
        data = dict(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        
        if self.test_mode:
            return data

        label = self.data["label"].iloc[index]
        label_id = torch.tensor(label,dtype=torch.float)
        data['label'] = label_id
        return data
    
    @classmethod
    def create_k_fold_dataloaders(cls, args):
        if args.k_fold > 1:
            print(f"进行{args.k_fold}折训练！")
            dataset = cls(args, args.train_path)
            skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
            # label = [dataset.data["Premise"].iloc[i] if args.premise else dataset.stance2label[dataset.data["Stance"].iloc[i]] for i in range(len(dataset))]
            label = [dataset.data["label"].iloc[i] for i in range(len(dataset))] 
            encoder = preprocessing.LabelEncoder() # 这两步是将label值映射成自增长的整型数字
            label = encoder.fit_transform(label)
            for train_idx, valid_idx in skf.split(dataset.data, label):
                train_dataset = Subset(dataset, train_idx)
                valid_dataset = Subset(dataset, valid_idx)

                train_sampler = RandomSampler(train_dataset) #定义采样器的策略
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
    def get_k_fold_testAndtrain_dataloader(cls, args):
        if args.k_fold > 1:
            print(f"进行{args.k_fold}折训练！")
            dataset = cls(args, args.train_path)
            
            label = [dataset.data["label"].iloc[i] for i in range(len(dataset))] 
            encoder = preprocessing.LabelEncoder() # 这两步是将label值映射成自增长的整型数字
            label = encoder.fit_transform(label)
            dataset.data["label_int"]=label
            
            #划分测试集
            X_index=[i for i in range(len(dataset))]
            X_train_index, X_test_index, label_train, label_test = train_test_split(
    X_index, label, test_size=0.2, stratify=label, random_state=1) 
            # stratify=label的意思：保持测试集的label与整个数据集里label的类别分布一致，分层采样。
            #创建一个test_dataloader
            test_dataset = Subset(dataset, X_test_index)
            test_sampler = SequentialSampler(test_dataset)
            train_data = dataset.data.iloc[X_train_index] #去掉测试集之后的新数据集
            train_dataset = cls(args, data=train_data)
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=args.val_batch_size,
                                            sampler=test_sampler,
                                            drop_last=False,
                                            pin_memory=True)
            print('The test data length: ',len(test_dataset))
            return test_dataloader, train_dataset
    
        
    
    @classmethod
    def create_k_fold_dataloaders1(cls, args, dataset):
        if args.k_fold > 1:
            skf = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
            label = [dataset.data["label_int"].iloc[i] for i in range(len(dataset))] 
            for train_idx, valid_idx in skf.split(dataset, label):
#                 train_idx1 = np.array(split_dataset.iloc[train_idx]['index'])
#                 valid_idx1 = np.array(split_dataset.iloc[valid_idx]['index'])
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
    def create_final_dataloader(cls, args, train_dataset, valid_dataset):
        if args.k_fold > 1:
            final_train_dataset = ConcatDataset([train_dataset, valid_dataset])
            final_train_sampler = RandomSampler(final_train_dataset)
            final_train_dataloader = DataLoader(final_train_dataset,
                                               batch_size=args.batch_size,
                                               sampler=final_train_sampler,
                                               drop_last=True,
                                               pin_memory=True)
#             print('The final_train data length: ',len(final_train_dataloader))
            return final_train_dataloader
    
    @classmethod
    def create_dataloaders(cls, args):
        
        data = pd.read_csv(args.train_path,sep=',')
        language_list=data.groupby("language").groups.keys()    
#         dev_language = sample(language_list,2)
#         dev_language = ['English','Portugeese']
        dev_language = ['Spanish','French']
        print("作为验证集的语种：",dev_language)
        dev_data = data[(data["language"]==dev_language[0]) | (data["language"]==dev_language[1])]
        train_data = data[(data["language"]!=dev_language[0]) & (data["language"]!=dev_language[1])]
        # print(dev_data.shape)
        train_dataset = cls(args, data=train_data)
        valid_dataset = cls(args, data=dev_data)
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
        # print(len(valid_dataloader.dataset))
        return train_dataloader, valid_dataloader
