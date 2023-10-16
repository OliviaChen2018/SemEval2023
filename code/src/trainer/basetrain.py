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
import pdb
from src.models import FocalMSE

class BaseTrainer:
    def __init__(self, args) -> None:
        self.args = args
        setup_logging()
        setup_device(args)
        setup_seed(args)
        self.get_dataloader()
        if args.k_fold > 1:
            self.k_fold_mse = []
            self.k_fold_pccs = []
            self.next_fold()
        os.makedirs(self.args.savedmodel_path, exist_ok=True)
        self.SetEverything(args)
        
    def SetEverything(self,args):
        self.args.best_score = 0 #baseline是0.8
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
        self.optimizer, self.scheduler = build_optimizer(args, self.model) # 设置优化器
        self.model.to(args.device) #将模型加载到device：GPU
        if self.args.swa_start > 0:
            print("使用SWA！")
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, args.swa_lr)
            self.swa_model.to(args.device)
        if self.args.ema_start >= 0:
            print("使用EMA！")
            self.ema = EMA(self.model, args.ema_decay)
            self.ema.register()
        
        self.resume()  # ckpt_file没有指定，则初始化self.start_epoch为0。
        if args.device == 'cuda':
            if args.distributed_train:
                print("多GPU训练!")
                self.model = torch.nn.parallel.DataParallel(self.model)
        if self.args.fgm != 0:
            logging.info("使用FGM进行embedding攻击！")
            self.fgm = FGM(self.model.module.roberta.embeddings.word_embeddings if hasattr(self.model, 'module') else \
                          self.model.roberta.embeddings.word_embeddings )
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
            # next() 返回迭代器的下一个项目，要和生成迭代器的 iter() 函数一起使用(在FTtrainer.py里)。
        
    def resume(self): 
        if self.args.ckpt_file is not None: # 按ckpt_file指定的路径加载上一个epoch训练保存的模型。

            checkpoint = torch.load(self.args.ckpt_file, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1 #所以设置key加1，用来保存新一轮训练的模型。
            print(f"load resume sucesses! epoch: {self.start_epoch - 1}, mean f1: {checkpoint['mean_f1']}")
        else: #ckpt_file没有指定
            self.start_epoch = 0
        
        
    def train(self): #train()是用来处理K折的，_train() 才是一个完整的训练流程。
        self.k = 0
        if self.args.k_fold <= 1:
            self._train() 
        else: #K折训练的代码
            save_root_path = self.args.savedmodel_path #设置保存模型的路径
            for k in range(1, self.args.k_fold+1): #几折交叉验证就遍历几遍
                # 保存模型的目录按照当前折数命名
                self.args.savedmodel_path = os.path.join(save_root_path,f"fold_{k}") 
                print(self.args.savedmodel_path)
                os.makedirs(self.args.savedmodel_path, exist_ok=True) #创建一个以savedmodel_path命名的目录
                self.k += 1
                logging.info(f'{self.k} fold starting......') #日志记录一下：第k折开始
                self._train()
                if self.k < self.args.k_fold:
                    self.k_fold_pccs.append(self.args.best_score)
                    self.next_fold()
                    self.SetEverything(self.args)
                else:
                # 这五步是另一种K折训练，最后取的是训练得到的K个模型的均值。所以本算法是在评估模型的效果，而不是在训练一个更好的模型。
#                     mse_loss = np.mean(self.k_fold_mse) 
#                     pccs = np.mean(self.k_fold_pccs)
                    #print(f'Final mse_loss is: {mse_loss}') #mse_loss因为不重要，所以没有记录均值
                    #print(f'Final pccs is: {pccs}')
                    #print(f'Final pccs is: {self.args.best_score}')
                    
                    #选出最好的一折的结果，使用全部训练集重新训练一次。
                    best_pccs_fold=self.k_fold_pccs.index(max(self.k_fold_pccs))+1
                    self.args.savedmodel_path=save_root_path
                    load_file_path = os.path.join(self.args.savedmodel_path,f"fold_{best_pccs_fold}") 
                    checkpoint = torch.load(f'{load_file_path}/final.bin') 
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logging.info(f'Choose the {best_pccs_fold}th fold, final training starting...')
                    self.k += 1 #?
                    self.get_final_dataloader()
                    self._train()
                    self.args.savedmodel_path=save_root_path
                    checkpoint = torch.load(f'{self.args.savedmodel_path}/final.bin') 
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    result = self.validate()
                    mse_loss = result["mseLoss"]
                    pccs = result["pearsonr"]
                    logging.info(f"current score is: {pccs}")
                    logging.info(f"mse loss is: {mse_loss}")
                    
        
    def _train(self):
        total_step = 0
        best_score = self.args.best_score # 每次训练开始时初始化best_score
        start_time = time.time()
        # num_total_steps = len(self.train_dataloader) * (self.args.max_epochs - self.start_epoch)
        #num_total_steps总步数=训练数据的条数×最大epoch轮数。num_total_steps用于打印剩余时间。
        num_total_steps = len(self.train_dataloader) * (self.args.max_epochs)  
        self.optimizer.zero_grad() # 每轮(epoch)训练之前将梯度清零
        for epoch in range(self.args.max_epochs):
            # 一轮(epoch)里面包含很多个batch的训练
            k = 0
            for single_step, batch in enumerate(tqdm(self.train_dataloader,desc="Training")):
                k = k+1
                self.model.train() 
                for key in batch:
                    batch[key] = batch[key].cuda()  #batch[key]是batch中的每一条数据
                if self.args.use_rdrop:
                    loss,acc = self.cal_rdrop_loss(batch)
                else:
#                     print(type(batch)) #batch是dict类型
                    loss,logits = self.model(batch) #调用model的forward函数
#                     loss, mse, logits = self.model(batch)
                if self.args.distributed_train:
                    loss = loss.mean()
                    # acc = acc.mean()
                loss.backward() # 反向传播
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
                self.optimizer.step() #参数更新
                self.optimizer.zero_grad() #每轮(epoch)训练之前要对梯度清零
                if self.args.ema_start >= 0 and total_step >= self.args.ema_start:
                    self.ema.update()
                if self.args.swa_start > 0:
                    self.swa_model.update_parameters(self.model)
                    self.swa_scheduler.step()
                else:
                    self.scheduler.step() #scheduler？
                
                total_step += 1 #total_step：当前执行的步数，即数据训练到多少条了，一条数据就是一步。
                if torch.isinf(logits).any() or torch.isnan(logits).any():
                        logging.info(f"第{total_step-total_step//epoch}条数据")
                        print(f'logits: {logits}')
                        print()
                        for name, parms in self.model.named_parameters():
                            print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
		 ' -->grad_value:',parms.grad)
                if total_step % self.args.print_steps == 0: #每print_steps步打印一次
                    time_per_step = (time.time() - start_time) / max(1, total_step) #time_per_step：每步需要的时间
                    remaining_time = time_per_step * (num_total_steps - total_step) #剩余时间是当前折的剩余时间
                    remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                    logging.info(f"Epoch {epoch}" 
                                    f"total_step {total_step}" 
                                    f"eta {remaining_time}:" 
                                    f"FocalMSE {loss:.4f}")
#                     logging.info(f"Epoch {epoch}" 
#                                     f"total_step {total_step}" 
#                                     f"eta {remaining_time}:" 
#                                     f"loss {loss:.4f}"
#                                     f"MSE {mse:.4f}" )
                    train_pccs = pearsonr(batch['label'].cpu(),logits.cpu().detach().numpy())[0] #打印一下训练集的pcc，看看有没有过拟合
                    logging.info(f"train_pccs {train_pccs}")
                    
                if total_step % self.args.save_steps == 0: # 每save_steps步计算一次指标，并保存模型。
                    if self.args.ema_start >= 0:
                        self.ema.apply_shadow()
                    result = self.validate()  # 这里的validate是为了打印训练过程中的模型效果
                    mse_loss = result["mseLoss"]
                    pccs = result["pearsonr"]
                    if self.args.ema_start >= 0:
                        self.ema.restore()
                    #如果当前batch的pccs比best_score更好，则更新模型状态和best_score。
                    #最后的best_score显示的是当前epoch的所有batch中效果最好的
                    if pccs > self.args.best_score or self.args.is_full: 
                        state = {
                                'epoch': epoch, 
                                'mse_loss': mse_loss,
                                "pccs":pccs,
                                } # 更新state中的mse_loss、pccs两个参数，使它总记录最好值。
                        if self.args.ema_start >= 0:
                            state['shadow'] = self.ema.shadow,
                            state['backup'] = self.ema.backup,
                        if self.args.distributed_train:
                            if self.args.swa_start > 0:
                                state['model_state_dict'] = self.swa_model.module.state_dict()
                            else:
                                state['model_state_dict'] = self.model.module.state_dict()
                        else:
                            state['model_state_dict'] = self.model.state_dict() # 调用state_dict()，在state中保存模型
#                             print(mse_loss)
#                         torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_pccs_{pccs[0]:.4f}_mse_{mse_loss:.4f}_{total_step}.bin') #模型保存到本地
                        torch.save(state, f'{self.args.savedmodel_path}/model_epoch_{epoch}_pccs_{pccs:.4f}_mse_{mse_loss:.4f}_{total_step}.bin') 
                        self.args.best_score = pccs
                        logging.info(f"best_score {self.args.best_score}")
                    logging.info(f"current_score {pccs}")
                        
            # Validation（一个epoch训练完成之后有个总的validate）
            if self.args.ema_start >= 0:
                self.ema.apply_shadow()
                
            result = self.validate()
            mse_loss = result["mseLoss"]
            pccs = result["pearsonr"]
            
            if self.args.ema_start >= 0:
                self.ema.restore()
            
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
                
            if pccs > self.args.best_score or self.args.is_full:
                self.args.best_score = pccs 
                # 每个epoch结束后，self.args.best_score中存放的都是本次epoch的最好结果。
            if self.args.k_fold>1:
                self.k_fold_mse.append(mse_loss)    #每个epoch结束后将模型当前的mse_loss记录到k_fold_mse中

            logging.info(f"best_pccs {self.args.best_score}")
            
            if epoch == self.args.max_epochs-1:
                torch.save(state, f'{self.args.savedmodel_path}/final.bin')
            if self.k>self.args.k_fold:
                logging.info(f"current_score {pccs}")
                  
    def validate(self):
        self.model.eval()
        if self.args.swa_start > 0:
            torch.optim.swa_utils.update_bn(self.train_dataloader, self.swa_model)
        predictions = []
        labels = []
        losses = []
        k_fold_predict = []
#         print(len(self.valid_dataloader)) # 70
#         print(len(self.valid_dataloader.dataset.dataset.data)) # 22279
        with torch.no_grad(): #跑验证集的时候相当于测试，不需要更新梯度。
            if self.k > self.args.k_fold:
                logging.info("使用测试集进行测试")
                dataloader = self.test_dataloader
#                 logging.info(f"the length of test_dataloader is: {len(dataloader)}")
            else:
                dataloader = self.valid_dataloader
#             dataloader = self.valid_dataloader
            for step, batch in enumerate(tqdm(dataloader,desc="Evaluating")):
                for k in batch:  # batch是dict类型
                    batch[k] = batch[k].cuda()
                if self.args.swa_start > 0:
                    loss, logits = self.swa_model(batch)
                else:
                    loss, logits = self.model(batch)
#                     loss, mse, logits = self.model(batch)
                if self.args.distributed_train:
                    logits = logits.mean()
                labels.extend(batch['label'].cpu().numpy())
                
                # print(logits.detach().cpu().numpy())
                predictions.extend(logits.detach().cpu().numpy())
        
        if self.args.k_fold > 1:
            if self.k <= self.args.k_fold:
	            valid_idx = self.valid_dataloader.dataset.indices
	            dev_data = self.valid_dataloader.dataset.dataset.data.iloc[valid_idx]
            else:# if-else和else的部分是新加的
                test_idx = self.test_dataloader.dataset.indices
                dev_data = self.test_dataloader.dataset.dataset.data.iloc[test_idx]
            #print(len(dev_data))
            dev_data["predict"]= predictions
        else:
            dev_data = self.valid_dataloader.dataset.data
            dev_data["predict"] = predictions
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
                