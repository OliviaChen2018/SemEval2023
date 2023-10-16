import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for COLING Shared Task 2022")
    parser.add_argument("--seed", type=int, default=42, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    
    # ========================= Data Configs ==========================
    parser.add_argument('--train_path', type=str, default='data/train.tsv')
    parser.add_argument('--valid_path', type=str, default='data/validation.tsv')
    parser.add_argument('--test_path', type=str, default='data/validation.tsv')
    
    
    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='data/checkpoint/')
    #parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--ckpt_file', type=str, default=None)
    parser.add_argument('--best_score', default=0.75, type=float, help='save checkpoint if mean_f1 > best_score')
    
    # ========================= Learning Configs ==========================
    parser.add_argument('--batch_size', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=64, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=64, type=int, help="use for testing duration per worker")
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    
    parser.add_argument('--max_steps', default=50000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--print_steps', type=int, default=100, help="Number of steps to log training metrics.")
    parser.add_argument('--warmup_steps', default=1500, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--warmup_rate', default=0.1, type=float, help="warm rate for parameters not in bert or vit")
    parser.add_argument('--gradient_accumulation_steps', default=2, type=int, help="gradient accumulation steps")
    
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    
    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='xlm-roberta-base')
    parser.add_argument('--bert_cache', type=str, default='data/bert_cache/')
    parser.add_argument('--bert_embedding_size', type=int, default=768, help='bert 的embedding size')
    parser.add_argument('--bert_hidden_size', type=int, default=768, help='bert 的output size')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    parser.add_argument('--bert_learning_rate', type=float, default=2e-5)
    parser.add_argument('--bert_warmup_steps', type=int, default=5000)
    parser.add_argument('--bert_max_steps', type=int, default=30000)
    parser.add_argument("--bert_hidden_dropout_prob", type=float, default=0.1)
    # ===========================GPU ID ===========================
    parser.add_argument('--gpu_ids', type=str, default='0')
    

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    # ===========================Mix up =========================
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--gamma', type=float, default=0)
    
    # ===========================对抗训练 =========================
    parser.add_argument('--fgm', type=int, default=0, help="大于0表示使用，为0则不使用")
    parser.add_argument('--pgd', type=int, default=0, help="大于0表示使用，为0则不使用")
    
    # ===========================rdrop =========================
    parser.add_argument('--use_rdrop', action='store_true', help='使用该参数表示使用R-drop')
    parser.add_argument('--rdrop_alpha', type=float, default=1.,help="KL loss的比例")
    
    # ===========================模型验证 =========================
    parser.add_argument('--save_steps', type=int, default=100, help="保存模型")
    # ===========================是否使用多卡训练 =========================
    parser.add_argument('--distributed_train', action='store_true', help='使用多卡进行训练')
    # ===========================任务标签类别 =========================
    parser.add_argument('--class_label', type=int, default=3, help="类别数量")
    # ===========================是否使用学习率不变 =========================
    parser.add_argument('--constant_lr', action='store_true', help='是否使用get_constant_schedule_with_warmup')
    
    # ===========================SWA =========================
    parser.add_argument('--swa_start', type=int, default=-1, help="SWA的起始epoch")
    parser.add_argument('--swa_lr', type=float, default=5e-5)
    # ===========================EMA =========================
    parser.add_argument('--ema_start', type=int, default=-1, help="EMA的起始step")
    parser.add_argument('--ema_decay', type=float, default=0.999)
    
    # =========================== 是否进行subtask b进行训练=========================
    parser.add_argument('--premise', action='store_true', help='是否进行premise分类')
    
    # =========================== 是否进行subtask b进行训练=========================
    parser.add_argument('--result_file', type=str, default='result.csv')
    # ===========================focal loss =========================
    parser.add_argument('--gamma_focal', type=float, default=-1,help="大于0则表示使用focal loss")
    
    # =========================== 是否使用持续预训练模型进行微调=========================
    parser.add_argument('--pretrain_model_path', type=str, default=None)

    # =========================== 是否进行对embedding进行冻结训练=========================
    parser.add_argument('--embedding_freeze', action='store_true', help='是否冻结embeddings')

    
    # =========================== 是否进行noisy tune=========================
    parser.add_argument('--noisy_tune', action='store_true', help='是否进行noisy tune进行初始化')
    
    parser.add_argument('--prob_path', type=str, default="probability.csv", help="类别数量")
    
    # =========================== 是否加入NSP任务=========================
    parser.add_argument('--NSP', action='store_true', help='是否加入NSP任务')
    # =========================== 是否multi_task任务=========================
    parser.add_argument('--multi_task', action='store_true', help='是否进行多任务训练')
    
    # =========================== 是否进行K折训练=========================
    parser.add_argument('--k_fold', type=int,default=3, help='是否进行K折训练')
    
    # =========================== Deberta微调参数=========================
    parser.add_argument('--is_deberta', action='store_true', help='是否利用Deberta进行微调')
    parser.add_argument('--deberta_dir', type=str, default="base")
    
    # =========================== K折推理根目录=========================
    parser.add_argument('--merge_root_path', type=str, default='data/')
    
    # =========================== K折推理根目录=========================
    parser.add_argument('--pseudo_data_path', type=str, default=None)
    
    
    parser.add_argument('--is_full', action='store_true', help='是否进行全量微调')
    
    
    
    return parser.parse_args()