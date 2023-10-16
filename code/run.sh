#!/bin/bash
echo "Starting training..."
python -m src.train \
    --gpu_ids='0,1,2,3,4,5,6'\
    --bert_seq_length=128\
    --learning_rate=1e-4\
    --bert_learning_rate=4e-5\
    --batch_size=16\
    --savedmodel_path='data/checkpoint/1015/test_notfold_SimMSE_trainAndtest_gamma5_dropout_epoch11'\
    --train_path="train_and_test.csv"\
    --max_epochs=11\
    --ema_start=-1\
    --ema_decay=0.99\
    --fgm=0\
    --bert_dir="xlm-roberta-base"\
    --class_label=1\
    --save_steps=100\
    --best_score=0.8\
    --k_fold=1
#     --rdrop_alpha=0.03 
    #--use_rdrop=1 \
    # --premise\
    # max_epochs是11
   # --savedmodel_path='data/checkpoint/1015/EP_5_fold_dropout_epoch11_gamma0.65_rdrop-alpha0.03'\
   #--batch_size=16\