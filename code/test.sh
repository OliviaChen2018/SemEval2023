#!/bin/bash
echo "Starting testing..."
python -m src.inference \
    --gpu_ids='0,1,2,3,4,5,6'\
    --bert_seq_length=128\
    --learning_rate=1e-4\
    --bert_learning_rate=4e-5\
    --batch_size=16\
    --test_path="semeval_test.csv"\
    --ema_start=-1\
    --ema_decay=0.99\
    --ckpt_file='/root/SemEval/code/data/checkpoint/1015/model_epoch_9_f1_0.3942_10900.bin'\
    --class_label=1\
    --savedmodel_path='data/checkpoint/1015'\
    --fgm=0\
    #ckpt_file的路径要写绝对路径才能找到，写相对路径会报错找不到文件

    #--bert_dir="xlm-roberta-base"\
    #--save_steps=100\
    #-best_score=0.8\
    #--k_fold=5
    # --premise\
    # --use_rdrop \
    # --rdrop_alpha 0.5 \
    # max_epochs是11
    
    # --max_epochs=11\