import os
import pickle
from datetime import datetime

import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from Linear_Model import Linear_Model
from Logger_morning import myLogger
from conf import conf, print_config
from feature_dataset import featureDataset, my_collate
from readData_fungo import read_fungo

args = conf()
if args.load_model is None or args.isTrain:
    comment = f'_{args.dataset}_{args.base_model}_{args.mode}_{args.remark}'
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + comment)
else:
    log_dir = ''.join(args.load_model[:args.load_model.rfind('/')])
    print(f'reuse dir: {log_dir}')
logger = myLogger(name='exp', log_path=log_dir + '.log')
# incompatible with logger...
writer = SummaryWriter(log_dir=log_dir)
writer.add_text('Parameters', str(vars(args)))
print_config(args, logger)
logger.setLevel(args.log_level)

from HMCN import HMCN
from HAN import HAN
from OHCNN_fast import OHCNN_fast
from TextCNN import TextCNN
from loadData import load_data_yelp, filter_ancestors, load_data_rcv1, split_multi, \
    load_data_rcv1_onehot, load_data_nyt_onehot, load_data_nyt
from model import Policy
from tree import Tree
from util import get_gpu_memory_map, save_checkpoint, check_doc_size, gen_minibatch_from_cache, gen_minibatch, \
    save_minibatch, contains_nan


def finish_episode(policy, update=True):
    policy_loss = []
    all_cum_rewards = []
    for i in range(args.n_rollouts):
        rewards = []
        R = np.zeros(len(policy.rewards[i][0]))
        for r in policy.rewards[i][::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        all_cum_rewards.extend(rewards)
        rewards = torch.Tensor(rewards)  # (length, batch_size)
        # logger.warning(f'original {rewards}')
        if args.baseline == 'avg':
            rewards -= policy.baseline_reward
        elif args.baseline == 'greedy':
            rewards_greedy = []
            R = np.zeros(len(policy.rewards_greedy[0]))
            for r in policy.rewards_greedy[::-1]:
                R = r + args.gamma * R
                rewards_greedy.insert(0, R)
            rewards_greedy = torch.Tensor(rewards_greedy)
            rewards -= rewards_greedy
        # logger.warning(f'after baseline {rewards}')
        if args.avg_reward_mode == 'batch':
            rewards = Variable((rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps)))
        elif args.avg_reward_mode == 'each':
            # mean/std is separate for each in the batch
            rewards = Variable((rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + float(np.finfo(np.float32).eps)))
        else:
            rewards = Variable(rewards)
        if args.gpu:
            rewards = rewards.cuda()
        for log_prob, reward in zip(policy.saved_log_probs[i], rewards):
            policy_loss.append(-log_prob * reward)
        # logger.warning(f'after mean_std {rewards}')
    if update:
        tree.n_update += 1
        try:
            policy_loss = torch.cat(policy_loss).mean()
        except Exception as e:
            logger.error(e)
        entropy = torch.cat(policy.entropy_l).mean()
        writer.add_scalar('data/policy_loss', policy_loss, tree.n_update)
        writer.add_scalar('data/entropy_loss', policy.beta * entropy.data[0], tree.n_update)
        policy_loss += policy.beta * entropy
        if args.sl_ratio > 0:
            policy_loss += args.sl_ratio * policy.sl_loss
        writer.add_scalar('data/sl_loss', args.sl_ratio * policy.sl_loss, tree.n_update)
        writer.add_scalar('data/total_loss', policy_loss.data[0], tree.n_update)
        optimizer.zero_grad()
        policy_loss.backward()
        if contains_nan(policy.class_embed.weight.grad):
            logger.error('nan in class_embed.weight.grad!')
        else:
            optimizer.step()
    policy.update_baseline(np.mean(np.concatenate(all_cum_rewards)))
    policy.finish_episode()


def calc_sl_loss(probs, doc_ids, update=True, return_var=False):
    mlb = MultiLabelBinarizer(classes=tree.class_idx)
    y_l = [tree.id2doc_ancestors[docid]['class_idx'] for docid in doc_ids]
    y_true = mlb.fit_transform(y_l)
    if args.gpu:
        y_true = Variable(torch.from_numpy(y_true)).cuda().float()
    else:
        y_true = Variable(torch.from_numpy(y_true)).float()
    loss = criterion(probs, y_true)
    if update:
        tree.n_update += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if return_var:
        return loss
    return loss.data[0]


def get_cur_size(tokens):
    if args.base_model != 'han':
        return tokens.size()[0]
    else:
        return tokens.size()[1]


def forward_step_sl(tokens, doc_ids, flat_probs_only=False):
    # TODO can reuse logits
    if args.global_ratio > 0:
        probs = policy.base_model(tokens, True)
        global_loss = calc_sl_loss(probs, doc_ids, update=False, return_var=True)
    else:
        probs = None
        global_loss = 0
    if flat_probs_only:
        policy.sl_loss = global_loss
        return global_loss, probs
    policy.doc_vec = None
    cur_batch_size = get_cur_size(tokens)
    cur_class_batch = np.zeros(cur_batch_size, dtype=int)
    for t in range(args.n_steps_sl):
        next_classes_batch = tree.p2c_batch(cur_class_batch)
        next_classes_batch_true, indices, next_class_batch_true, doc_ids = tree.get_next(cur_class_batch,
                                                                                         next_classes_batch,
                                                                                         doc_ids)
        policy.step_sl(tokens, cur_class_batch, next_classes_batch, next_classes_batch_true)
        cur_class_batch = next_class_batch_true
        policy.duplicate_doc_vec(indices)
    policy.sl_loss /= args.n_steps
    writer.add_scalar('data/step-sl_sl_loss', (1 - args.global_ratio) * policy.sl_loss, tree.n_update)
    policy.sl_loss = (1 - args.global_ratio) * policy.sl_loss + args.global_ratio * global_loss
    writer.add_scalar('data/flat_sl_loss', args.global_ratio * global_loss, tree.n_update)
    return global_loss, probs


def train_step_sl():
    policy.train()
    for i in range(1, args.num_epoch + 1):
        g = select_data('train' + args.ohcnn_data, shuffle=True)
        loss_total = 0
        tree.cur_epoch = i
        for ct, (tokens, doc_ids) in tqdm(enumerate(g)):
            if 'FUN' in args.dataset or 'GO' in args.dataset:
                tokens = Variable(tokens).cuda()
            global_loss, flat_probs = forward_step_sl(tokens, doc_ids, flat_probs_only=(args.global_ratio == 1))
            optimizer.zero_grad()
            policy.sl_loss.backward()
            optimizer.step()
            tree.n_update += 1
            loss_total += policy.sl_loss.data[0]
            if ct % args.output_every == 0 and ct != 0:
                if args.global_ratio > 0:
                    logger.info(
                        f'loss_cur:{policy.sl_loss.data[0]} global_loss:{args.global_ratio * global_loss.data[0]}')
                else:
                    logger.info(f'loss_cur:{policy.sl_loss.data[0]} global_loss:off')
                logger.info(f'[{i}:{ct}] loss_avg:{loss_total / ct}')
                writer.add_scalar('data/sl_loss', global_loss, tree.n_update)
                writer.add_scalar('data/loss_avg', loss_total / ct, tree.n_update)
            policy.sl_loss = 0
        if i % args.save_every == 0:
            eval_save_model(i, datapath='train' + args.ohcnn_data, save=True, output=False)
            test_step_sl('test' + args.ohcnn_data, save_prob=False)


def test_step_sl(data_path, save_prob=False, output=True):
    logger.info('test starts')
    policy.eval()
    g = select_data(data_path)
    pred_l = []
    target_l = []
    for ct, (tokens, doc_ids) in tqdm(enumerate(g)):
        if 'FUN' in args.dataset or 'GO' in args.dataset:
            tokens = Variable(tokens).cuda()
        real_doc_ids = [i for i in doc_ids]
        policy.doc_vec = None
        cur_batch_size = get_cur_size(tokens)
        cur_class_batch = np.zeros(cur_batch_size, dtype=int)
        for _ in range(args.n_steps_sl):
            next_classes_batch = tree.p2c_batch(cur_class_batch)
            probs = policy.step_sl(tokens, cur_class_batch, next_classes_batch, None, sigmoid=True)
            indices, next_class_batch_pred, doc_ids = tree.get_next_by_probs(cur_class_batch, next_classes_batch,
                                                                             doc_ids, probs, save_prob)
            cur_class_batch = next_class_batch_pred
            policy.duplicate_doc_vec(indices)
        last_did = None
        for c, did in zip(cur_class_batch, doc_ids):
            if last_did != did:
                pred_l.append([])
            if c != 0:
                pred_l[-1].append(c)
            last_did = did
        target_l.extend(real_doc_ids)
    if save_prob:
        logger.info(f'saving {writer.file_writer.get_logdir()}/{data_path}.tree.id2prob2.pkl')
        pickle.dump(tree.id2prob, open(f'{writer.file_writer.get_logdir()}/{data_path}.tree.id2prob2.pkl', 'wb'))
        tree.id2prob.clear()
        return
    return evaluate(pred_l, target_l, output=output)


def output_log(cur_class_batch, doc_ids, acc, i, ct):
    if args.dataset in ['yelp', 'rcv1']:
        label_key = 'categories'
        try:
            logger.info('pred{}{} real{}{} pred_h{} real_h{}'.format(cur_class_batch[:3],
                                                                     [tree.id2name[tree.idx2id[cur]] for cur in
                                                                      cur_class_batch[:3]],
                                                                     [tree.id2doc_ancestors[docid]['class_idx'] for
                                                                      docid in doc_ids[:3]],
                                                                     [tree.id2doc_ancestors[docid][label_key] for docid
                                                                      in doc_ids[:3]],
                                                                     tree.h_batch(cur_class_batch[:3]),
                                                                     tree.h_doc_batch(doc_ids[:3])))
        except Exception as e:
            logger.warning(e)
    writer.add_scalar('data/acc', np.mean(acc), tree.n_update)
    writer.add_scalar('data/beta', policy.beta, tree.n_update)
    logger.info('single-label acc for epoch {} batch {}: {}'.format(i, ct, np.mean(acc)))
    if (cur_class_batch == cur_class_batch[0]).all():
        logger.error('predictions in a batch are all the same! [{}]'.format(cur_class_batch[0]))
        writer.add_text('error', 'predictions in a batch are all the same! [{}]'.format(cur_class_batch[0]),
                        tree.n_update)
        if not args.debug:
            exit(1)


def train_taxo():
    policy.train()
    for i in range(1, args.num_epoch + 1):
        g = select_data('train' + args.ohcnn_data, shuffle=True)
        pred_l = []
        target_l = []
        tree.cur_epoch = i
        for ct, (tokens, doc_ids) in tqdm(enumerate(g)):
            if 'FUN' in args.dataset or 'GO' in args.dataset:
                tokens = Variable(tokens).cuda()
            flat_probs = None
            if args.sl_ratio > 0:
                _, flat_probs = forward_step_sl(tokens, doc_ids, flat_probs_only=(args.global_ratio == 1))
                if not args.mix_flat_probs:
                    flat_probs = None
            policy.doc_vec = None
            cur_batch_size = get_cur_size(tokens)

            # greedy baseline
            if args.baseline == 'greedy':
                tree.taken_actions = [set() for _ in range(cur_batch_size)]
                cur_class_batch = np.zeros(cur_batch_size, dtype=int)
                next_classes_batch = [set() for _ in range(cur_batch_size)]
                for t in range(args.n_steps):
                    next_classes_batch, next_classes_batch_np, _ = tree.get_next_candidates(cur_class_batch,
                                                                                            next_classes_batch)
                    choices, m = policy.step(tokens, cur_class_batch, next_classes_batch_np, test=True,
                                             flat_probs=flat_probs)
                    cur_class_batch = next_classes_batch_np[
                        np.arange(len(next_classes_batch_np)), choices.data.cpu().numpy()]
                    tree.update_actions(cur_class_batch)
                    policy.rewards_greedy.append(tree.calc_reward(t < args.n_steps - 1, cur_class_batch, doc_ids))
                tree.last_R = None
            for _ in range(args.n_rollouts):
                policy.saved_log_probs.append([])
                policy.rewards.append([])
                tree.taken_actions = [set() for _ in range(cur_batch_size)]
                cur_class_batch = np.zeros(cur_batch_size, dtype=int)
                next_classes_batch = [set() for _ in range(cur_batch_size)]
                for t in range(args.n_steps):
                    next_classes_batch, next_classes_batch_np, all_stop = tree.get_next_candidates(cur_class_batch,
                                                                                                   next_classes_batch)
                    if args.early_stop and all_stop:
                        break
                    choices, m = policy.step(tokens, cur_class_batch, next_classes_batch_np, flat_probs=flat_probs)
                    cur_class_batch = next_classes_batch_np[
                        np.arange(len(next_classes_batch_np)), choices.data.cpu().numpy()]
                    tree.update_actions(cur_class_batch)
                    policy.saved_log_probs[-1].append(m.log_prob(choices))
                    policy.rewards[-1].append(tree.calc_reward(t < args.n_steps - 1, cur_class_batch, doc_ids))
                tree.last_R = None
                tree.remove_stop()
                pred_l.extend(tree.taken_actions)
                target_l.extend(doc_ids)
            finish_episode(policy, update=True)
            if ct % args.output_every == 0 and ct != 0:
                logger.info(f'epoch {i} batch {ct}')
                eval_save_model(i, pred_l, target_l, output=False)
        if i % args.save_every == 0:
            eval_save_model(i, pred_l, target_l, save=True, output=False)
            test_taxo('test' + args.ohcnn_data, save_prob=False)


def select_data(data_path, shuffle=False):
    if 'FUN' in args.dataset or 'GO' in args.dataset:
        isTrain = False
        if 'train' in data_path:
            isTrain = True
        test_dataset = featureDataset(args.dataset, isTrain)
        g = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=my_collate, shuffle=shuffle)
    elif args.base_model == 'ohcnn-bow-fast':
        g = gen_minibatch_from_cache(logger, args, tree, args.batch_size, name=data_path, shuffle=shuffle)
    else:
        if 'test' in data_path:
            g = gen_minibatch(logger, args, word_index, X_test, test_ids, args.batch_size, shuffle=shuffle)
        else:
            g = gen_minibatch(logger, args, word_index, X_train, train_ids, args.batch_size, shuffle=shuffle)
    return g


def test_taxo(data_path, save_prob=False):
    logger.info('test starts')
    policy.eval()
    g = select_data(data_path)
    pred_l = []
    target_l = []
    if save_prob:
        args.n_steps = tree.n_class - 1
    for ct, (tokens, doc_ids) in tqdm(enumerate(g)):
        if 'FUN' in args.dataset or 'GO' in args.dataset:
            tokens = Variable(tokens).cuda()
        flat_probs = None
        if args.sl_ratio > 0 and args.mix_flat_probs:
            _, flat_probs = forward_step_sl(tokens, doc_ids, flat_probs_only=True)
        policy.doc_vec = None
        cur_batch_size = get_cur_size(tokens)
        tree.taken_actions = [set() for _ in range(cur_batch_size)]
        cur_class_batch = np.zeros(cur_batch_size, dtype=int)
        next_classes_batch = [set() for _ in range(cur_batch_size)]
        for _ in range(args.n_steps):
            next_classes_batch, next_classes_batch_np, all_stop = tree.get_next_candidates(cur_class_batch,
                                                                                           next_classes_batch,
                                                                                           save_prob)
            if all_stop:
                if save_prob:
                    logger.error('should not enter')
                break
            choices, m = policy.step(tokens, cur_class_batch, next_classes_batch_np, test=True, flat_probs=flat_probs)
            cur_class_batch = next_classes_batch_np[
                np.arange(len(next_classes_batch_np)), choices.data.cpu().numpy()]
            if save_prob:
                for did, idx, p_ in zip(doc_ids, cur_class_batch,
                                        m.probs.gather(-1, choices.unsqueeze(-1)).squeeze(-1).data.cpu().numpy()):
                    assert 0 < idx < 104, idx
                    if idx in tree.id2prob[did]:
                        logger.warning(f'[{did}][{idx}] already existed!')
                    tree.id2prob[did][idx] = p_
            tree.update_actions(cur_class_batch)
        tree.remove_stop()
        pred_l.extend(tree.taken_actions)
        target_l.extend(doc_ids)
    if save_prob:
        logger.info(f'saving {writer.file_writer.get_logdir()}/{data_path}.tree.id2prob-rl.pkl')
        pickle.dump(tree.id2prob, open(f'{writer.file_writer.get_logdir()}/{data_path}.tree.id2prob-rl.pkl', 'wb'))
        tree.id2prob.clear()
    return evaluate(pred_l, target_l)


def eval_save_model(i, pred_l=None, target_l=None, datapath=None, save=False, output=True):
    if args.mode == 'hilap-sl':
        test_f = test_step_sl
    elif args.mode == 'hilap':
        test_f = test_taxo
    else:
        test_f = test_sl
    if pred_l:
        f1, f1_a, f1_aa, f1_macro, f1_a_macro, f1_aa_macro, f1_aa_s = evaluate(pred_l, target_l, output=output)
    elif datapath:
        f1, f1_a, f1_aa, f1_macro, f1_a_macro, f1_aa_macro, f1_aa_s = test_f(datapath, output=output)
    else:
        f1, f1_a, f1_aa, f1_macro, f1_a_macro, f1_aa_macro, f1_aa_s = test_f(X_train, train_ids)
    writer.add_scalar('data/micro_train', f1_aa, tree.n_update)
    writer.add_scalar('data/macro_train', f1_aa_macro, tree.n_update)
    writer.add_scalar('data/samples_train', f1_aa_s, tree.n_update)
    if not save:
        return
    if args.mode in ['hilap', 'hilap-sl']:
        save_checkpoint({
            'state_dict': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, writer.file_writer.get_logdir(), f'epoch{i}_{f1_aa}_{f1_aa_macro}_{f1_aa_s}.pth.tar', logger, True)
    else:
        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, writer.file_writer.get_logdir(), f'epoch{i}_{f1_aa}_{f1_aa_macro}_{f1_aa_s}.pth.tar', logger, True)


def decode(pred_l, target_l, doc_ids, probs):
    cur_class_batch = None
    target_l.extend(doc_ids)
    if args.multi_label:
        if args.mode != 'hmcn':
            probs = torch.sigmoid(probs)
        preds = (probs >= .5).int().data.cpu().numpy()
        for pred in preds:
            idx = np.nonzero(pred)[0]
            if len(idx) == 0:
                pred_l.append([])
            else:
                pred_l.append(idx + 1)
    else:
        cur_class_batch = torch.max(probs, 1)[1].data.cpu().numpy() + 1
        pred_l.extend(cur_class_batch)
    return pred_l, target_l, cur_class_batch


def train_hmcn():
    model.train()
    for i in range(1, args.num_epoch + 1):
        g = select_data('train' + args.ohcnn_data, shuffle=True)
        loss_total = 0
        pred_l = []
        target_l = []
        for ct, (tokens, doc_ids) in tqdm(enumerate(g)):
            if 'FUN' in args.dataset or 'GO' in args.dataset:
                tokens = Variable(tokens).cuda()
            probs = model(tokens)
            pred_l, target_l, cur_class_batch = decode(pred_l, target_l, doc_ids, probs)
            loss = calc_sl_loss(probs, doc_ids, update=True)
            if ct % 50 == 0 and ct != 0:
                logger.info('loss: {}'.format(loss))
                if args.multi_label:
                    acc = tree.acc_multi(np.array(pred_l), np.array(target_l))
                else:
                    acc = tree.acc(np.array(pred_l), np.array(target_l))
                logger.info('acc for epoch {} batch {}: {}'.format(i, ct, acc))
                writer.add_scalar('data/loss', loss, tree.n_update)
                writer.add_scalar('data/acc', acc, tree.n_update)
                if not args.multi_label and (cur_class_batch == cur_class_batch[0]).all():
                    logger.error('predictions in a batch are all the same! [{}]'.format(cur_class_batch[0]))
                    writer.add_text('error', 'predictions in a batch are all the same! [{}]'.format(cur_class_batch[0]),
                                    tree.n_update)
                    # exit(1)
            loss_total += loss
        loss_avg = loss_total / (ct + 1)
        if args.multi_label:
            acc = tree.acc_multi(np.array(pred_l), np.array(target_l))
        else:
            acc = tree.acc(np.array(pred_l), np.array(target_l))
        logger.info('loss_avg:{} acc:{}'.format(loss_avg, acc))
        if not args.multi_label:
            pred_l = [[label] for label in pred_l]
        if i % args.save_every == 0:
            eval_save_model(i, pred_l, target_l, save=True)


def test_hmcn():
    logger.info('testing starts')
    model.eval()
    g = select_data('test' + args.ohcnn_data)
    loss_total = 0
    pred_l = []
    target_l = []
    probs_l = []
    for ct, (tokens, doc_ids) in tqdm(enumerate(g)):
        if 'FUN' in args.dataset or 'GO' in args.dataset:
            tokens = Variable(tokens).cuda()
        probs = model(tokens)
        probs_l.append(probs.data.cpu().numpy())
        pred_l, target_l, cur_class_batch = decode(pred_l, target_l, doc_ids, probs)
        loss = calc_sl_loss(probs, doc_ids, update=False)
        loss_total += loss
    probs_np = np.concatenate(probs_l, axis=0)
    logger.info('saving probs to {}/probs_{}.pkl'.format(writer.file_writer.get_logdir(), tree.n_update))
    pickle.dump((probs_np, target_l),
                open('{}/probs_{}.pkl'.format(writer.file_writer.get_logdir(), tree.n_update), 'wb'))
    loss_avg = loss_total / (ct + 1)
    if args.multi_label:
        acc = tree.acc_multi(np.array(pred_l), np.array(target_l))
    else:
        acc = tree.acc(np.array(pred_l), np.array(target_l))
    logger.info('loss_avg:{} acc:{}'.format(loss_avg, acc))
    if not args.multi_label:
        pred_l = [[label] for label in pred_l]
    inc = 0
    for p in pred_l:
        p = set(p)
        exist = False
        for l in p:
            cur = tree.c2p_idx[l][0]
            while cur != 0:
                if cur not in p:
                    inc += 1
                    exist = True
                    break
                cur = tree.c2p_idx[cur][0]
            if exist:
                break
    print(inc)
    # exit()
    return evaluate(pred_l, target_l)


def train_sl():
    model.train()
    for i in range(1, args.num_epoch + 1):
        g = select_data('train' + args.ohcnn_data, shuffle=True)
        loss_total = 0
        pred_l = []
        target_l = []
        for ct, (tokens, doc_ids) in tqdm(enumerate(g)):
            probs = model(tokens, True)
            pred_l, target_l, cur_class_batch = decode(pred_l, target_l, doc_ids, probs)
            loss = calc_sl_loss(probs, doc_ids, update=True)
            if ct % args.output_every == 0 and ct != 0:
                logger.info('sl_loss: {}'.format(loss))
                if args.multi_label:
                    acc = tree.acc_multi(np.array(pred_l), np.array(target_l))
                else:
                    acc = tree.acc(np.array(pred_l), np.array(target_l))
                logger.info('acc for epoch {} batch {}: {}'.format(i, ct, acc))
                writer.add_scalar('data/sl_loss', loss, tree.n_update)
                writer.add_scalar('data/acc', acc, tree.n_update)
                if not args.multi_label and (cur_class_batch == cur_class_batch[0]).all():
                    logger.error('predictions in a batch are all the same! [{}]'.format(cur_class_batch[0]))
                    writer.add_text('error', 'predictions in a batch are all the same! [{}]'.format(cur_class_batch[0]),
                                    tree.n_update)
                    # exit(1)
            loss_total += loss
        loss_avg = loss_total / (ct + 1)
        if args.multi_label:
            acc = tree.acc_multi(np.array(pred_l), np.array(target_l))
        else:
            acc = tree.acc(np.array(pred_l), np.array(target_l))
        logger.info('loss_avg:{} acc:{}'.format(loss_avg, acc))
        if not args.multi_label:
            pred_l = [[label] for label in pred_l]
        if i % args.save_every == 0:
            # eval_save_model(i, pred_l, target_l, save=True)
            test_sl()


def test_sl():
    logger.info('testing starts')
    model.eval()
    g = select_data('test' + args.ohcnn_data)
    loss_total = 0
    pred_l = []
    target_l = []
    probs_l = []
    for ct, (tokens, doc_ids) in tqdm(enumerate(g)):
        probs = model(tokens, True)
        probs_l.append(probs.data.cpu().numpy())
        pred_l, target_l, cur_class_batch = decode(pred_l, target_l, doc_ids, probs)
        loss = calc_sl_loss(probs, doc_ids, update=False)
        loss_total += loss
    # probs_np = np.concatenate(probs_l, axis=0)
    # logger.info('saving probs to {}/probs_{}.pkl'.format(writer.file_writer.get_logdir(), tree.n_update))
    # pickle.dump((probs_np, target_l),
    #             open('{}/probs_{}.pkl'.format(writer.file_writer.get_logdir(), tree.n_update), 'wb'))
    loss_avg = loss_total / (ct + 1)
    if args.multi_label:
        acc = tree.acc_multi(np.array(pred_l), np.array(target_l))
    else:
        acc = tree.acc(np.array(pred_l), np.array(target_l))
    logger.info('loss_avg:{} acc:{}'.format(loss_avg, acc))
    if not args.multi_label:
        pred_l = [[label] for label in pred_l]
    return evaluate(pred_l, target_l)


def evaluate(pred_l, test_ids, save_path=None, output=True):
    acc = round(tree.acc_multi(pred_l, test_ids), 4)
    res = tree.calc_f1(pred_l, test_ids, save_path, output)
    if output:
        f1, f1_a, f1_aa, f1_macro, f1_a_macro, f1_aa_macro, f1_aa_s = [round(i, 4) for i in res]
        logger.info(
            f'acc:{acc} f1_s:{f1_aa_s} micro-f1:{f1} {f1_a} {f1_aa} macro-f1:{f1_macro} {f1_a_macro} {f1_aa_macro}')
        if f1_aa > tree.miF[0]:
            tree.miF = (f1_aa, f1_aa_macro, tree.cur_epoch)
        if f1_aa_macro > tree.maF[1]:
            tree.maF = (f1_aa, f1_aa_macro, tree.cur_epoch)
        logger.warning(f'best: {tree.miF}, {tree.maF}')
        return f1, f1_a, f1_aa, f1_macro, f1_a_macro, f1_aa_macro, f1_aa_s
    else:
        f1_a, f1_a_macro, f1_a_s = [round(i, 4) for i in res]
        return 0, 0, f1_a, 0, 0, f1_a_macro, f1_a_s


if args.dataset == 'rcv1':
    if 'oh' in args.base_model:
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = load_data_rcv1_onehot('_rcv1_ptAll')
    else:
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = load_data_rcv1(
            '../datasets/glove.6B.50d.txt', '_rcv1_ptAll')
    if args.filter_ancestors:
        id2doc_train = filter_ancestors(id2doc, nodes)
        if args.split_multi:
            X_train, train_ids, id2doc_train, id2doc = split_multi(X_train, train_ids, id2doc_train, id2doc)
    else:
        id2doc_train = id2doc
    tree = Tree(args, train_ids, test_ids, id2doc=id2doc_train, id2doc_a=id2doc, nodes=nodes, rootname='Root')

elif args.dataset == 'yelp':
    subtree_name = 'root'
    min_reviews = 5
    max_reviews = 10
    X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = load_data_yelp('../datasets/glove.6B.50d.txt',
                                                                                         '_yelp_root_100_{}_{}'.format(
                                                                                             min_reviews, max_reviews),
                                                                                         root=subtree_name,
                                                                                         min_reviews=min_reviews,
                                                                                         max_reviews=max_reviews)
    logger.warning(f'{len(X_train)} {len(train_ids)} {len(X_test)} {len(test_ids)}')
    # save_minibatch(logger, args, word_index, X_train, train_ids, 32, name='train_yelp_root_100_5_10_len256_padded')
    # save_minibatch(logger, args, word_index, X_test, test_ids, 32, name='test_yelp_root_100_5_10_len256_padded')
    if args.filter_ancestors:
        id2doc_train = filter_ancestors(id2doc, nodes)
    else:
        id2doc_train = id2doc
    tree = Tree(args, train_ids, test_ids, id2doc=id2doc_train, id2doc_a=id2doc, nodes=nodes, rootname=subtree_name)
elif args.dataset == 'nyt':
    if 'oh' in args.base_model:
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = load_data_nyt_onehot('_nyt_ptAll')
    else:
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = load_data_nyt(
            '../datasets/glove.6B.50d.txt', '_nyt_ptAll')
    if args.filter_ancestors:
        id2doc_train = filter_ancestors(id2doc, nodes)
        if args.split_multi:
            X_train, train_ids, id2doc_train, id2doc = split_multi(X_train, train_ids, id2doc_train, id2doc)
    else:
        id2doc_train = id2doc
    save_minibatch(logger, args, word_index, X_train, train_ids, 32, name='train_nyt')
    save_minibatch(logger, args, word_index, X_test, test_ids, 32, name='test_nyt')
    tree = Tree(args, train_ids, test_ids, id2doc=id2doc_train, id2doc_a=id2doc, nodes=nodes, rootname='Top')
elif 'FUN' in args.dataset or 'GO' in args.dataset:
    X_train, _, train_ids, test_ids, id2doc, nodes = read_fungo(args.dataset)
    if args.filter_ancestors:
        id2doc_train = filter_ancestors(id2doc, nodes)
    else:
        id2doc_train = id2doc
    tree = Tree(args, train_ids, test_ids, id2doc=id2doc_train, id2doc_a=id2doc, nodes=nodes, rootname='Top')
else:
    logger.error('No such dataset: {}'.format(args.dataset))
    exit(1)
if args.stat_check:
    check_doc_size(X_train, logger)
    check_doc_size(X_test, logger)
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    sorted_gpu_info = get_gpu_memory_map()
    for gpu_id, (mem_left, util) in sorted_gpu_info:
        if mem_left >= args.min_mem:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info('use gpu:{} with {} MB left, util {}%'.format(gpu_id, mem_left, util))
            break
    else:
        logger.warn(f'no gpu has memory left >= {args.min_mem} MB, exiting...')
        exit()
else:
    torch.set_num_threads(10)
if 'cnn' in args.base_model:
    if args.base_model == 'textcnn':
        model = TextCNN(args, word_vec=wv, n_classes=tree.n_class - 1)
        in_dim = 3000
    elif args.base_model == 'ohcnn-bow-fast':
        model = OHCNN_fast(word_index['UNK'], n_classes=tree.n_class - 1, vocab_size=len(word_index))
        in_dim = 10000
    if args.mode == 'hmcn':
        local_output_size = tree.get_layer_node_number()
        model = HMCN(model, args, local_output_size, tree.n_class - 1, in_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    if args.gpu:
        logger.info(model.cuda())
elif args.base_model == 'han':
    model = HAN(args, word_vec=wv, n_classes=tree.n_class - 1)
    in_dim = args.sent_gru_hidden_size * 2
    if args.mode == 'sl':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    if args.gpu:
        logger.info(model.cuda())
elif args.base_model == 'raw':
    local_output_size = tree.get_layer_node_number()
    if args.dataset == 'rcv1':
        model = HMCN(None, args, local_output_size, tree.n_class - 1, 47236)
    elif args.dataset == 'yelp':
        model = HMCN(None, args, local_output_size, tree.n_class - 1, 146587)
    elif args.dataset == 'nyt':
        model = HMCN(None, args, local_output_size, tree.n_class - 1, 102755)
    elif 'FUN' in args.dataset or 'GO' in args.dataset:
        n_features = len(X_train[0])
        logger.info(f'n_features={n_features}')
        if args.mode == 'hmcn':
            model = HMCN(None, args, local_output_size, tree.n_class - 1, n_features)
        else:
            model = Linear_Model(args, n_features, tree.n_class - 1)
            X_train, X_test, train_ids, test_ids = None, None, None, None
            in_dim = args.n_hidden
    if args.gpu:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_weight)

base_model = model
if args.mode in ['hilap', 'hilap-sl']:
    if args.mode == 'hilap':
        policy = Policy(args, tree.n_class + 1, base_model, in_dim)
    else:
        policy = Policy(args, tree.n_class, base_model, in_dim)
    if args.gpu:
        logger.info(policy.cuda())
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    for name, param in policy.named_parameters():
        logger.info('{} {} {}'.format(name, type(param.data), param.size()))

if args.mode == 'hmcn':
    criterion = torch.nn.BCELoss()
else:
    criterion = torch.nn.BCEWithLogitsLoss()
if args.load_model:
    if os.path.isfile(args.load_model):
        checkpoint = torch.load(args.load_model)
        load_optimizer = True
        if args.mode in ['sl', 'hmcn']:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            policy_dict = policy.state_dict()
            load_from_sl = False
            if load_from_sl:
                for i in list(checkpoint['state_dict'].keys()):
                    checkpoint['state_dict']['base_model.' + i] = checkpoint['state_dict'].pop(i)
                checkpoint['state_dict']['class_embed.weight'] = torch.cat(
                    [policy_dict['class_embed.weight'][-1:], checkpoint['state_dict']['base_model.fc2.weight'],
                     policy_dict['class_embed.weight'][-1:]])
                checkpoint['state_dict']['class_embed_bias.weight'] = torch.cat(
                    [policy_dict['class_embed_bias.weight'][-1:],
                     checkpoint['state_dict']['base_model.fc2.bias'].view(-1, 1),
                     policy_dict['class_embed_bias.weight'][-1:]])
                load_optimizer = False
            elif checkpoint['state_dict']['class_embed.weight'].size()[0] == \
                    policy_dict['class_embed.weight'].size()[0] - 1:
                logger.warning('try loading pretrained x for rl-taxo. class_embed also loaded.')
                load_optimizer = False
                # checkpoint['state_dict']['class_embed.weight'] = policy_dict['class_embed.weight']
                checkpoint['state_dict']['class_embed.weight'] = torch.cat(
                    [checkpoint['state_dict']['class_embed.weight'], policy_dict['class_embed.weight'][-1:]])
                checkpoint['state_dict']['class_embed_bias.weight'] = torch.cat(
                    [checkpoint['state_dict']['class_embed_bias.weight'], policy_dict['class_embed_bias.weight'][-1:]])
                logger.warning(checkpoint['state_dict']['class_embed.weight'].size())
            policy.load_state_dict(checkpoint['state_dict'])
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            logger.warning('optimizer not loaded')
        logger.info("loaded checkpoint '{}' ".format(args.load_model))
    else:
        logger.error("no checkpoint found at '{}'".format(args.load_model))
        exit(1)
if args.stat_check:
    evaluate([[tree.id2idx[id2doc_train[did]['categories'][0]]] for did in train_ids], train_ids)
    evaluate([[tree.id2idx[id2doc_train[did]['categories'][0]]] for did in test_ids], test_ids)
if not args.load_model or args.isTrain:
    if args.mode == 'hilap-sl':
        train_step_sl()
        # test_tmp('test' + args.ohcnn_data)
        test_step_sl('test' + args.ohcnn_data, save_prob=False)
    elif args.mode == 'hilap':
        train_taxo()
        # test_taxo('train' + args.ohcnn_data, save_prob=False)
        # test_taxo('test' + args.ohcnn_data, save_prob=False)
    elif args.mode == 'hmcn':
        train_hmcn()
    else:
        train_sl()
        test_sl()
else:
    if args.mode == 'hilap':
        # test_step_sl('train' + args.ohcnn_data, save_prob=False)
        # test_step_sl('test' + args.ohcnn_data, save_prob=True)
        test_taxo('train' + args.ohcnn_data, save_prob=False)
        test_taxo('test' + args.ohcnn_data, save_prob=False)
    elif args.mode == 'hilap-sl':
        # test_sl(X_test, test_ids)  # for testing global_flat with additional local_loss
        # test_step_sl('train' + args.ohcnn_data, save_prob=True)
        # test_tmp('test' + args.ohcnn_data)
        test_step_sl('train' + args.ohcnn_data, save_prob=False)
        test_step_sl('test' + args.ohcnn_data, save_prob=False)
    elif args.mode == 'hmcn':
        test_hmcn()
    else:
        test_sl()
writer.close()
logger.info(f'log_dir: {log_dir}')
