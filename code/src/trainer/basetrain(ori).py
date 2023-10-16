import logging
import os
import time
import torch
from tqdm import tqdm
from src.config import parse_args
from src.dataset import FinetuneDataset
# from src.
# from src.models import MultiModal
from src.utils import *
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from torch.optim.swa_utils import AveragedModel, SWALR
# from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

class BaseTrainer:
    def __init__(self, args) -> None:
        self.args = args
        setup_logging()
        setup_device(args)
        setup_seed(args)
        self.get_dataloader()
        if args.k_fold > 1:
            self.next_fold()
        os.makedirs(self.args.savedmodel_path, exist_ok=True)
        self.SetEverything(args)
        
    def SetEverything(self,args):
        self.args.best_score = 0.8
        self.get_model()
        ######是否冻结embedding层
        if args.embedding_freeze:
            freeze(self.model.roberta.embeddings)
            print("冻结word embedding参数不训练!")
            
        #####是否进行noisytune
        if args.noisy_tune:
            print("#######Nosiy Tune##########")
            noisyT=NoisyTune(self.model)
            noisyT.add_noisy(0.2)
        
        args.max_steps = args.max_epochs * len(self.train_dataloader)
        args.warmup_steps = int(args.warmup_rate * args.max_steps)
        self.optimizer, self.scheduler = build_optimizer(args, self.model)
        self.model.to(args.device)
        if self.args.swa_start > 0:
            print("使用SWA！")
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, args.swa_lr)
            self.swa_model.to(args.device)
        if self.args.ema_start >= 0:
            print("使用EMA！")
            self.ema = EMA(self.model, args.ema_decay)
            self.ema.register()
        
        self.resume()
        if args.device == 'cuda':
            if args.distributed_train:
                print("多GPU训练!")
                self.model = torch.nn.parallel.DataParallel(self.model)
        if self.args.fgm != 0:
            logging.info("使用FGM进行embedding攻击！")
            self.fgm = FGM(self.model.module.roberta.embeddings.word_embeddings if hasattr(self.model, 'module') else \
                          self.model.roberta.embeddings.word_embeddings
                          )
        if self.args.pgd != 0:
            # self.pgd = PGD(self.model)
            pass
        
        logging.info("Training/evaluation parameters: %s", args)

        
    def get_model(self):
        raise NotImplementedError('you need implemented this function')
    
    def get_dataloader(self):
        # self.train_dataloader, self.valid_dataloader = FinetuneDataset.create_dataloaders(self.args)
        raise NotImplementedError('you need implemented this function')
        
        
    def next_fold(self):
        if self.args.k_fold > 1:
            self.train_dataloader, self.valid_dataloader = next(self.dataloaders)
        
    def resume(self):
        if self.args.ckpt_file is not None:

            checkpoint = torch.load(self.args.ckpt_file, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"load resume sucesses! epoch: {self.start_epoch - 1}, mean f1: {checkpoint['mean_f1']}")
        else:
            self.start_epoch = 0
        
        
    def train(self):
        self.k = 0
        if self.args.k_fold <= 1:
            self._train()
        else:
            save_root_path = self.args.savedmodel_path
            for k in range(1, self.args.k_fold+1):
                self.args.savedmodel_path = os.path.join(save_root_path,f"fold_{k}")
                print(self.args.savedmodel_path)
                os.makedirs(self.args.savedmodel_path, exist_ok=True)
                self.k += 1
                logging.info(f'{self.k} fold starting......')
                self._train()
                if self.k < self.args.k_fold:
                    self.next_fold()
                    self.SetEverything(self.args)
        
    def _train(self):
        total_step = 0
        best_score = self.args.best_score
        start_time = time.time()
        # num_total_steps = len(self.train_dataloader) * (self.args.max_epochs - self.start_epoch)
        num_total_steps = len(self.train_dataloader) * (self.args.max_epochs)
        self.optimizer.zero_grad()
        for epoch in range(self.args.max_epochs):
            for single_step, batch in enumerate(tqdm(self.train_dataloader,desc="Training:")):
                self.model.train()
                for key in batch:
                    batch[key] = batch[key].cuda()
                if self.args.use_rdrop:
                    loss,acc = self.cal_rdrop_loss(batch)
                else:
                    loss,logits = self.model(batch)
                if self.args.distributed_train:
                    loss = loss.mean()
                    # acc = acc.mean()
                loss.backward()
                if self.args.fgm !=0 :
                    self.fgm.attack(0.5+epoch * 0.1)
                    if self.args.use_rdrop:
                        loss_adv, _ = self.cal_rdrop_loss(batch)
                    else:
                        loss_adv, _ = self.model(batch)
                    loss_adv = loss_adv.mean()
                    loss_adv.backward()
                    self.fgm.restore()
                if self.args.pgd !=0 :
                    pass # 后续需要再补充
                # if self.args.multi_task:
                #     torch.nn.utils.clip_grad_norm_(self.model.stance_cls.parameters(), self.args.max_grad_norm)
                #     torch.nn.utils.clip_grad_norm_(self.model.premise_cls.parameters(), self.args.max_grad_norm)
                # else:
                #     torch.nn.utils.clip_grad_norm_(self.model.module.cls.parameters() if hasattr(self.model, "module") else self.model.cls.parameters(), self.args.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.args.ema_start >= 0 and total_step >= self.args.ema_start:
                    self.ema.update()
                if self.args.swa_start > 0:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                else:
                    self.scheduler.step()
                
                total_step += 1
                if total_step % self.args.print_steps == 0:
                    time_per_step = (time.time() - start_time) / max(1, total_step)
                    remaining_time = time_per_step * (num_total_steps - total_step)
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch}" 
                                    f"total_step {total_step}" 
                                    f"eta {remaining_time}:" 
                                    f"MSE loss {loss:.4f}")
                    
                if total_step % self.args.save_steps == 0:
                    if self.args.ema_start >= 0:
                        self.ema.apply_shadow()
                    result = self.validate()
                    mse_loss = result["mseLoss"]
                    pccs = result["pearsonr"]
                    if self.args.ema_start >= 0:
                        self.ema.restore()
                    if pccs > self.args.best_score or self.args.is_full:
                        state = {
                                'epoch': epoch, 
                                'mse_loss': mse_loss,
                                "pccs":pccs,
                                }
                        if self.args.ema_start >= 0:
                            state['shadow'] = self.ema.shadow,
                            state['backup'] = self.ema.backup,
                        if self.args.distributed_train:
                            if self.args.swa_start > 0:
                                state['model_state_dict'] = self.swa_model.module.state_dict()
                            else:
                                state['model_state_dict'] = self.model.module.state_dict()
                        else:
                            state['model_state_dict'] = self.model.state_dict()
                        torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_f1_{mse_loss:.4f}_{total_step}.bin')
                        self.args.best_score = pccs
                        logging.info(f"best_score {self.args.best_score}")
                    logging.info(f"current_score {pccs}")
                        
            # Validation
            if self.args.ema_start >= 0:
                self.ema.apply_shadow()
            result = self.validate()
            mse_loss = result["mseLoss"]
            pccs = result["pearsonr"]
            
            if self.args.ema_start >= 0:
                self.ema.restore()
            if pccs > self.args.best_score or self.args.is_full:
                state = {
                        'epoch': epoch, 
                        'mse_loss': mse_loss,
                        "pccs":pccs,
                        }
                if self.args.ema_start >= 0:
                    state['shadow'] = self.ema.shadow,
                    state['backup'] = self.ema.backup,
                if self.args.distributed_train:
                    if self.args.swa_start > 0:
                        state['model_state_dict'] = self.swa_model.module.state_dict()
                    else:
                        state['model_state_dict'] = self.model.module.state_dict()
                else:
                    state['model_state_dict'] = self.model.state_dict()
                torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_f1_{mse_loss:.4f}_{total_step}.bin')
                self.args.best_score = pccs
                logging.info(f"best_mse {self.args.best_score}")
                
                
    def validate(self):
        self.model.eval()
        if self.args.swa_start > 0:
            torch.optim.swa_utils.update_bn(self.train_dataloader, self.swa_model)
        predictions = []
        labels = []
        losses = []
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.valid_dataloader,desc="Evaluating")):
                for k in batch:
                    batch[k] = batch[k].cuda()
                if self.args.swa_start > 0:
                    loss, logits = self.swa_model(batch)
                else:
                    loss, logits = self.model(batch)
                if self.args.distributed_train:
                    logits = logits.mean()
                labels.extend(batch['label'].cpu().numpy())
                # print(logits.detach().cpu().numpy())
                predictions.extend(logits.detach().cpu().numpy())
        
        dev_data = self.valid_dataloader.dataset.data
        dev_data["predict"]= predictions
        dev_data.to_csv("data/dev.csv",index=None,sep=",")
        mseLoss = mean_squared_error(labels,predictions)
        pccs = pearsonr(labels,predictions)[0]
        result = dict(
            mseLoss = mseLoss,
            pearsonr = pccs,
        )
        return result
        
    def cal_rdrop_loss(self, batch):
        loss1, logits1, accuracy1, _ = self.model(batch)
        loss2, logits2, accuracy2, _ = self.model(batch)
        loss1 = loss1.mean()
        loss2 = loss2.mean()
        ce_loss  = (loss1+loss2) * 0.5
        accuracy = (accuracy1+accuracy2) * 0.5
        kl_loss = compute_kl_loss(logits1, logits2)
        loss = ce_loss + self.args.rdrop_alpha * kl_loss
        return loss, accuracy
                