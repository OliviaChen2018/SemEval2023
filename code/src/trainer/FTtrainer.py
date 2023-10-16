from .basetrain import BaseTrainer
from src.models import FtModel
from src.config import parse_args
from src.dataset import FinetuneDataset
class FtTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
    def get_dataloader(self):
        if self.args.k_fold > 1:
            print("进行K折训练！")
            self.test_dataloader, self.train_dataset = FinetuneDataset.get_k_fold_testAndtrain_dataloader(self.args)#新加的
            # self.dataloaders其实是一个以FinetuneDataset.create_k_fold_dataloaders1为参数的迭代器
            self.dataloaders = iter(FinetuneDataset.create_k_fold_dataloaders1(self.args, self.train_dataset))
            #self.dataloaders = iter(FinetuneDataset.create_k_fold_dataloaders(self.args)) #原来只有这一条
            print(f"训练集的条数为: {len(self.train_dataset)}")
        else:
            self.train_dataloader, self.valid_dataloader = FinetuneDataset.create_dataloaders(self.args)
        
    def get_final_dataloader(self):
        if self.args.k_fold > 1 and self.k > self.args.k_fold:
            train_dataset = self.train_dataloader.dataset
            valid_dataset = self.valid_dataloader.dataset
            self.train_dataloader = FinetuneDataset.create_final_dataloader(self.args, train_dataset, valid_dataset)
            print(f"训练集的条数为: {len(self.train_dataloader.dataset)}")
    def get_model(self):
        self.model = FtModel(self.args)
        
        
        
        
