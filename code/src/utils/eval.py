from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


def class_f1(labels_df,is_premise):
    """
    根据官网的评测规则确定线上评测
    把预测的标签加入验证集，并命名为Predict
    """
    group_df = labels_df.groupby("Claim")
    all_f1 = []
    for claim, index in group_df.groups.items():
        label_claim = labels_df["Premise"].iloc[index]  if is_premise else labels_df["Stance"].iloc[index] 
        predict_claim = labels_df["Predict"].iloc[index] 
        
        f1_micro = f1_score(label_claim, predict_claim, average='micro')
        f1_macro = f1_score(label_claim, predict_claim, average='macro')
        all_f1.append((f1_micro+f1_macro)/2.)
    return sum(all_f1)/len(all_f1)