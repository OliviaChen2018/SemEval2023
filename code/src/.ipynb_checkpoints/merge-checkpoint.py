import os
import pandas as pd
import numpy as np
import argparse
from tqdm import trange

def merge(args):
    prob_paths = ["data/stance_probability_twitter_v2_B_full_1300.csv","data/stance_probability_twitter_v2_B_full_1400.csv"]
    label2stance = {"0":"AGAINST","1":"FAVOR","2":"NONE"}
    prob_final = None
    for path in prob_paths:
        prob_df = pd.read_csv(path,header=None,sep="\t")
        # print(prob_df)
        if prob_final is not None:
            prob_final = prob_final + prob_df
        else:
            prob_final = prob_df
    final_label = []
    for i in trange(prob_final.shape[0]):
        label_id = np.argmax(prob_final.iloc[i])
        if args.premise:
            label = label_id
        else:
            label = label2stance[str(label_id)]
        final_label.append(label)
    final_prediction = pd.read_csv(args.prediction_path,sep="\t")
    if args.premise:
        final_prediction["Premise"] = final_label
    else:
        final_prediction["Stance"] = final_label
    if args.premise:
        result_path = os.path.join(os.path.split(args.prediction_path)[0],"premise_merge.tsv")
    else:
        result_path = os.path.join(os.path.split(args.prediction_path)[0],"stance_merge.tsv")
        
    final_prediction.to_csv(result_path, sep="\t",index=None)
def main():
    parser = argparse.ArgumentParser(description="Baseline for COLING Shared Task 2022")
    parser.add_argument('--prediction_path', type=str, default='data/stance_result_twitter_v2_B_full_1300.tsv')
    parser.add_argument('--premise', action='store_true', help='是否进行premise分类')
    
    args = parser.parse_args()
    merge(args)
if __name__=="__main__":
    main()
    
    
        
    