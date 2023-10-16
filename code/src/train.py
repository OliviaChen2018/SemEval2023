from src.config import parse_args
from src.trainer import FtTrainer
import torch
import os
if __name__ == "__main__":
    args = parse_args()
    print("The GPUS ids: ",args.gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
    torch.cuda.set_device(2)
    trainer = FtTrainer(args)
    trainer.train()