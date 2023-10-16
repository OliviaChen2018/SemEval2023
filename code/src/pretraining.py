import src.pretrain_cfg as args
from src.trainer import PretrainTrainer
import os
if __name__ == "__main__":
    print("The GPUS ids: ",args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    trainer = PretrainTrainer(args)
    trainer.train()