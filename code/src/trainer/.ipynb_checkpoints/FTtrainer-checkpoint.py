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
            self.dataloaders = iter(FinetuneDataset.create_k_fold_dataloaders(self.args))
        else:
            self.train_dataloader, self.valid_dataloader = FinetuneDataset.create_dataloaders(self.args)
        
    def get_model(self):
        self.model = FtModel(self.args)
        
        
        
        
