import argparse


def print_config(config, logger=None):
    config = vars(config)
    info = "Running with the following configs:\n"
    for k, v in config.items():
        info += "\t{} : {}\n".format(k, str(v))
    if not logger:
        print("\n" + info + "\n")
    else:
        logger.info("\n" + info + "\n")


def conf():
    ap = argparse.ArgumentParser()
    # change the parameters to test different method/base_model/dataset combinations
    ap.add_argument('--mode', default='sl', choices=['rl-taxo', 'sl', 'step-sl', 'hmcn'])
    ap.add_argument('--base_model', default='textcnn', choices=['han', 'textcnn', 'ohcnn-bow-fast', 'raw'])
    ap.add_argument('--dataset', default='rcv1', choices=['yelp', 'rcv1', 'nyt', 'cellcycle_FUN'])
    ap.add_argument('--isTrain', default=False, help='True for continuing training')
    ap.add_argument('--load_model', default=None)
    ap.add_argument('--remark', default='del', help='reminder of this run')

    # most of following parameters do not need any changes
    ap.add_argument('--lr', default=1e-3, help='learning rate 1e-3 for ohcnn, textcnn, 1e-1 for han')
    ap.add_argument('--l2_weight', default=1e-6, help='weight decay of optimizer')
    ap.add_argument('--save_every', default=10, help='evaluate and save model every k epochs')
    ap.add_argument('--num_epoch', default=50)
    ap.add_argument('--word_gru_hidden_size', default=50)
    ap.add_argument('--sent_gru_hidden_size', default=50)
    ap.add_argument('--hist_embed_size', default=50)
    ap.add_argument('--update_beta_every', default=500)
    ap.add_argument('--pretrained_word_embed', default=True)
    ap.add_argument('--update_word_embed', default=True)
    ap.add_argument('--allow_stay', default=True,
                    help='if sample_mode=random, has to be False in case select prob=0->nan')
    ap.add_argument('--sample_mode', default='normal', choices=['choose_max', 'random', 'normal'])
    ap.add_argument('--batch_size', default=32)
    ap.add_argument('--batch_size_test', default=32)
    ap.add_argument('--log_level', default=20)
    ap.add_argument('--stat_check', default=False, help='calculate and print some stats of the data')
    ap.add_argument('--max_tokens', default=256, help='max size of tokens')
    ap.add_argument('--debug', default=False, help='if True, run some epochs on the FIRST batch')
    ap.add_argument('--gamma', default=.9, help='discounting factor')
    ap.add_argument('--beta', default=2, help='weight of entropy')
    ap.add_argument('--use_cur_class_embed', default=True, help='add embedding of current class to state embedding')
    ap.add_argument('--use_l1', default=True)
    ap.add_argument('--use_l2', default=True, help='only valid when use_l1=True')
    ap.add_argument('--l1_size', default=500, help='output size of l1. only valid when use_l2=True')
    ap.add_argument('--class_embed_size', default=50)
    ap.add_argument('--softmax', default=True, choices='softmax or sigmoid')
    ap.add_argument('--sl_ratio', default=1, help='[0 = off] for rl-taxo: sl_loss = rl_loss + sl_ratio * sl_loss')
    ap.add_argument('--global_ratio', default=0.5,
                    help='[0 = off]for step-sl: sl_loss = (1-global_ratio) * local_loss + global_ratio * global_loss')
    ap.add_argument('--gpu', default=True)
    ap.add_argument('--n_rollouts', default=1)
    ap.add_argument('--reward', default='f1', choices=['01', '01-1', 'f1', 'direct', 'taxo'])
    ap.add_argument('--early_stop', default=False, help='for rl-taxo only')
    ap.add_argument('--baseline', default='greedy', choices=[None, 'avg', 'greedy'])
    ap.add_argument('--avg_reward_mode', default='batch', choices=['off', 'each', 'batch'],
                    help='if n_step=1, cannot be each ->nan')
    ap.add_argument('--min_mem', default=3000, help='minimum gpu memory requirement (MB)')
    # outdated parameters
    ap.add_argument('--allow_up', default=False, help='not used anymore')
    ap.add_argument('--use_history', default=False, help='not used anymore')
    ap.add_argument('--multi_label', default=True, help='whether predict multi labels. valid for sl and step-sl')
    ap.add_argument('--split_multi', default=False,
                    help='split one sample with m labels to m samples, only valid when filter_ancestors=True')
    ap.add_argument('--mix_flat_probs', default=False, help='add flat prob to help rl')
    args = ap.parse_args()
    args.use_history &= (args.mode == 'rl')
    if args.dataset == 'rcv1':
        args.ohcnn_data = '_rcv1_len256_padded'  # where to load ohcnn cache
        args.n_steps_sl = 4  # steps of step-sl
        args.n_steps = 17  # steps of rl
        args.output_every = 723  # output metrics every k batches
    elif args.dataset == 'yelp':
        args.ohcnn_data = '_yelp_root_100_5_10_len256_padded'
        args.n_steps_sl = 4
        args.n_steps = 10
        args.output_every = 2730
    elif args.dataset == 'nyt':
        args.ohcnn_data = '_nyt'
        args.n_steps_sl = 3
        args.n_steps = 20
        args.output_every = 789
    elif 'FUN' in args.dataset:
        args.ohcnn_data = '_cellcycle_FUN_root_padded'
        args.n_steps_sl = 6
        args.n_steps = 45
        args.output_every = 100
        args.n_hidden = 150
        args.l1_size = 1000
        args.class_embed_size = 1000
        args.dropout = 0
    elif 'GO' in args.dataset:
        args.ohcnn_data = '_cellcycle_GO_root_padded'
        args.n_steps_sl = 14
        args.n_steps = 45
        args.output_every = 100
        args.n_hidden = 150
        args.l1_size = 1000
        args.class_embed_size = 1000
        args.dropout = 0
    if args.n_steps == 1:
        args.avg_reward_mode = 'off'
    if args.mode in ['sl', 'hmcn']:
        args.filter_ancestors = False  # 'only use lowest labels as gold for training'
    else:
        args.filter_ancestors = True
    return args
