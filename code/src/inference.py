from src.models import FtModel
from src.config import parse_args
from src.dataset import FinetuneDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset, WeightedRandomSampler
import os
from tqdm import tqdm,trange
import pandas as pd
import torch
from src.utils import *
def inference():
    args = parse_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print("The GPUS ids: ",args.gpu_ids)
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6'
    setup_device(args)
    setup_seed(args)
    torch.cuda.set_device(2)
    test_dataset = FinetuneDataset(args, args.test_path, True)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=test_sampler,
                                    drop_last=False,
                                    pin_memory=True)
    
    print('The test data length: ',len(test_dataloader))
    model = FtModel(args)
    model = model.to(args.device)
    if args.distributed_train:
        model = torch.nn.parallel.DataParallel(model)
    print(args.ckpt_file)
    ckpoint = torch.load(args.ckpt_file)
    model.load_state_dict(ckpoint['model_state_dict'])
    #print("The epoch {} and the best mean f1 {:.4f} of the validation set.".format(ckpoint['epoch'],ckpoint['mean_f1']))
    
    if args.ema_start >= 0:
        ema = EMA(model, args.ema_decay)
        ema.resume(ckpoint['shadow'][0], ckpoint['backup'][0])
        # ema.shadow = 
        ema.apply_shadow()
    
    model.eval() 
    predict = []
    probs_list = np.empty((0,args.class_label),dtype=float)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_dataloader,desc="Evaluating")):
            for k in batch:
                batch[k] = batch[k].cuda() 
                #batch 存放的具体是什么？打印一下看看
            probability = model(batch,True) #传参：inference为True，网络前向传播只计算score。
            #pred_label_id = torch.argmax(probability, dim=1) #取logits最大的那个类别作为预测类
            #predictions.extend(pred_label_id.cpu().numpy())
            probs_list = np.concatenate([probs_list,probability.cpu().detach().numpy()])
    # 保存预测概率进行模型融合
    prob_df = pd.DataFrame(probs_list)
    # prob_path参数默认值为'probability.csv'。将prob_df写入probability.csv。
    #prob_df.to_csv(os.path.join("data",args.prob_path),sep="\t",header=None,index=None)
    df = pd.read_csv("data/sem_result.csv")
    df["predictions"] = prob_df
    df.to_csv("data/sem_final_result.csv")
#     # 写入预测文件
#     with open(f"data/{args.result_file}","w+") as f:
#         if args.premise:
#             f.write(f"text\tlanguage\tpredictions\n")
#         else:
#             f.write(f"text\tlanguage\tpredictions\n")
#         for i in trange(len(probs_list)):
#             #i_d = test_dataset.data['id'].iloc[i]
#             text = test_dataset.data['text'].iloc[i]
#             #claim = test_dataset.data['Claim'].iloc[i]
#             language=test_dataset.data['language'].iloc[i]
#             if args.premise:
#                 predictions = int(predict[i])
#             else:
#                 #label = predictions[i]
#                 predictions=probs_list[i][0]
#             # label = int(predictions[i])
#             f.write(f"{text}\t{language}\t{predictions}\n") #将三个变量：text、language和predict写入result.csv
if __name__ == "__main__":
    inference()