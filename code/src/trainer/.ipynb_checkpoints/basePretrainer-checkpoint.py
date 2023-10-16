import logging
import os
import time
import torch
from tqdm import tqdm
from src.dataset import PretrainDataset
# from src.
# from src.models import MultiModal
from src.utils import *
from sklearn.metrics import f1_score, accuracy_score
from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings
warnings.filterwarnings("ignore")

class BaseTrainer:
    def __init__(self, args) -> None:
        self.args = args
        setup_logging()
        setup_device(args)
        setup_seed(args)
        self.SetEverything(args)
        
    def SetEverything(self,args):
        self.get_model()
        self.get_dataloader()
        args.max_steps = args.max_epochs * len(self.train_dataloader)
        args.warmup_steps = int(args.warmup_rate * args.max_steps)
        self.optimizer, self.scheduler = build_pretrain_optimizer(args, self.model)
        self.model.to(args.device) 
        self.resume()
        if args.device == 'cuda':
            if args.distributed_train:
                print("多GPU训练!")
                self.model = torch.nn.parallel.DataParallel(self.model)
        os.makedirs(self.args.savedmodel_path, exist_ok=True)
        logging.info("Training/evaluation parameters: %s", args)

        
    def get_model(self):
        raise NotImplementedError('you need implemented this function')
    
    def get_dataloader(self):
        # self.train_dataloader, self.valid_dataloader = FinetuneDataset.create_dataloaders(self.args)
        raise NotImplementedError('you need implemented this function')
        
        
    def resume(self):
        if self.args.ckpt_file is not None:

            checkpoint = torch.load(self.args.ckpt_file, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"load resume sucesses! epoch: {self.start_epoch - 1}, loss: {checkpoint['loss']}")
        else:
            self.start_epoch = 0
        
        
    def train(self):
        total_step = 0
        best_loss = self.args.best_loss
        start_time = time.time()
        num_total_steps = len(self.train_dataloader) * (self.args.max_epochs)
        self.optimizer.zero_grad()
        for epoch in range(self.args.max_epochs):
            for single_step, batch in enumerate(tqdm(self.train_dataloader,desc="PreTraining:")):
                self.model.train()
                for key in batch:
                    batch[key] = batch[key].cuda()
                loss_all= self.model(batch)
                loss = loss_all["loss"]
                loss_mlm = loss_all["mlm_loss"]
                loss_nsp = loss_all["nsp_loss"]
                
                if self.args.distributed_train:
                    loss = loss.mean()
                    loss_mlm = loss_mlm.mean()
                    loss_nsp = loss_nsp.mean()
                    
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                total_step += 1
                if total_step % self.args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, total_step)
                    remaining_time = time_per_step * (num_total_steps - total_step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch}" 
                                    f"total_step {total_step}" 
                                    f"eta {remaining_time}:" 
                                    f"loss_all {loss:.3f}"
                                    f"loss_mlm {loss_mlm:.3f}"
                                    f"loss_nlp {loss_nsp:.3f}"
                                )
                    
                if total_step % self.args.save_steps == 0:
                    loss = self.validate()
                    loss_mlm = loss["mlm_loss"]
                    loss_nsp = loss["nsp_loss"]
                    loss_all = loss["loss"]
                    if loss_all < self.args.best_loss:
                        state = {
                                'epoch': epoch, 
                                'loss': loss_all,
                                }
                        if self.args.distributed_train:
                            if "roberta" in self.args.bert_dir:
                                state['model_state_dict'] = self.model.module.mlm.roberta.state_dict()
                            else:
                                state['model_state_dict'] = self.model.module.mlm.bert.state_dict()
                        else:
                            if "roberta" in self.args.bert_dir:
                                state['model_state_dict'] = self.model.mlm.roberta.state_dict()
                            else:
                                state['model_state_dict'] = self.model.mlm.bert.state_dict()
                        torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_loss_{loss_all:.3f}_{total_step}_{loss_mlm:.3f}_{loss_nsp:.3f}.bin')
                        self.args.best_loss = loss_all
                        logging.info(f"best_loss {self.args.best_loss}")
                    logging.info(f"current_loss {loss}")
                        
            # Validation
            loss = self.validate()
            loss_mlm = loss["mlm_loss"]
            loss_nsp = loss["nsp_loss"]
            loss_all = loss["loss"]
            
            if loss_all < self.args.best_loss or True:
                state = {
                        'epoch': epoch, 
                        'loss': loss_all,
                        }
                if self.args.distributed_train:
                    if "roberta" in self.args.bert_dir:
                        state['model_state_dict'] = self.model.module.mlm.roberta.state_dict()
                    else:
                        state['model_state_dict'] = self.model.module.mlm.bert.state_dict()
                else:
                    if "roberta" in self.args.bert_dir:
                        state['model_state_dict'] = self.model.mlm.roberta.state_dict()
                    else:
                        state['model_state_dict'] = self.model.mlm.bert.state_dict()
                torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_loss_{loss_all:.3f}_{total_step}_{loss_mlm:.3f}_{loss_nsp:.3f}.bin')
                if loss_all < self.args.best_loss:
                    self.args.best_loss = loss_all
                logging.info(f"best_loss {self.args.best_loss}")
                
    def validate(self):
        self.model.eval()
        losses = []
        losses_mlm = []
        losses_nsp = []
        
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.valid_dataloader,desc="PretrainEvaluating")):
                for k in batch:
                    batch[k] = batch[k].cuda()
                if self.args.swa_start > 0:
                    loss_all = self.swa_model(batch)
                else:
                    loss_all = self.model(batch)
                loss = loss_all["loss"]
                loss = loss.mean()
                losses.append(loss.cpu().numpy())
                if "mlm_loss" in loss_all.keys():
                    loss_mlm = loss_all["mlm_loss"]
                    loss_nsp = loss_all["nsp_loss"]

                    loss_mlm = loss_mlm.mean()
                    loss_nsp = loss_nsp.mean()

                    losses_mlm.append(loss_mlm.cpu().numpy())
                    losses_nsp.append(loss_nsp.cpu().numpy())
                
        loss = sum(losses) / len(losses)
        loss_mlm = sum(losses_mlm) / len(losses_mlm) if len(losses_mlm) > 0 else 0
        loss_nsp = sum(losses_nsp) / len(losses_nsp) if len(losses_nsp) > 0 else 0
        
        loss_dict = dict(
                mlm_loss = loss_mlm,
                nsp_loss = loss_nsp,
                loss = loss
            )
        return loss_dict
        

                    
                