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
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    setup_device(args)
    setup_seed(args)
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
        
    # args.merge_root_path = "data/checkpoint/0715_ema_6_stance_twitter_rdrop_v2_fold/"
    model_list = os.listdir(args.merge_root_path)
    
    prediction_list = []
    probability_merge = np.zeros((len(test_dataset),args.class_label),dtype=float)
    
    for idx, model_name in enumerate(model_list):
        print(f"Preparing to load the {idx+1}th checkpoint ! The model name:{model_name}")
        model_path = os.path.join(args.merge_root_path, model_name)
        ckpoint = torch.load(model_path)
        model.load_state_dict(ckpoint['model_state_dict'])
        print("The epoch {} and the best mean f1 {:.4f} of the validation set.".format(ckpoint['epoch'],ckpoint['mean_f1']))
    
        if args.ema_start >= 0:
            ema = EMA(model, args.ema_decay)
            ema.resume(ckpoint['shadow'][0], ckpoint['backup'][0])
            # ema.shadow = 
            ema.apply_shadow()
    
        model.eval()
        predictions = []
        probs_list = np.empty((0,args.class_label),dtype=float)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(test_dataloader,desc="Evaluating")):
                for k in batch:
                    batch[k] = batch[k].cuda()

                probability = model(batch,True)
                pred_label_id = torch.argmax(probability, dim=1)
                predictions.extend(pred_label_id.cpu().numpy())
                probs_list = np.concatenate([probs_list,probability.cpu().detach().numpy()])
        prediction_list.append(predictions)
        probability_merge += probs_list
       
    ##两种融合策略，概率融合，投票
    probability_merge_result = [np.argmax(probability_merge[i]) for i in range(probability_merge.shape[0])]
    # vote
    prediction_array  = np.array(prediction_list)
    prediction_array = prediction_array.T
    vote_results = []
    for i in range(prediction_array.shape[0]):
        item = prediction_array[i]
        freq_count = np.bincount(item)
        vote = item[np.argmax(freq_count)]
        vote_results.append(vote)
    # 保存预测概率进行模型融合
    # prob_df = pd.DataFrame(probs_list)
    # prob_df.to_csv(os.path.join("data",args.prob_path),sep="\t",header=None,index=None)
    # 写入预测文件
    save_path_name_list = ["prob_merge_B.tsv","vote_merge_B.tsv"]
    for idx,save_path in enumerate(save_path_name_list):
        if idx == 0:
            predictions = probability_merge_result
        else:
            predictions = vote_results
        with open(f"data/{save_path}","w+") as f:
            if args.premise:
                f.write(f"id\ttext\tClaim\tPremise\n")
            else:
                f.write(f"id\ttext\tClaim\tStance\n")
            for i in trange(len(predictions)):
                i_d = test_dataset.data['id'].iloc[i]
                text = test_dataset.data['Tweet'].iloc[i]
                claim = test_dataset.data['Claim'].iloc[i]
                if args.premise:
                    label = int(predictions[i])
                else:
                    label = test_dataset.label2stance[str(int(predictions[i]))]
                # label = int(predictions[i])

                f.write(f"{i_d}\t{text}\t{claim}\t{label}\n")
if __name__ == "__main__":
    inference()