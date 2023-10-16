from .basePretrainer import BaseTrainer
from src.models import PretrainRoberta
from src.dataset import PretrainDataset
class PretrainTrainer(BaseTrainer):
    def __init__(self, args) -> None:
        super().__init__(args)
        
    def get_dataloader(self):
        self.train_dataloader, self.valid_dataloader = PretrainDataset.create_dataloaders(self.args)
        
    def get_model(self):
        self.model = PretrainRoberta(self.args)
        