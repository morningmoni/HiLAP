import collections
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm


def isnan(x):
    return x != x


def contains_nan(x):
    return isnan(x).any()


def explode(x):
    return (x > 10).any()


def eu_dist(x):
    return sum((x[0] - x[1]) ** 2) / len(x[0])


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_info = [eval(x) for x in result.strip().split('\n')]
    gpu_info = dict(zip(range(len(gpu_info)), gpu_info))
    sorted_gpu_info = sorted(gpu_info.items(), key=lambda kv: kv[1][0], reverse=True)
    sorted_gpu_info = sorted(sorted_gpu_info, key=lambda kv: kv[1][1])
    print(f'gpu_id, (mem_left, util): {sorted_gpu_info}')
    return sorted_gpu_info


def save_checkpoint(state, modelpath, modelname, logger=None, del_others=True):
    if del_others:
        for dirpath, dirnames, filenames in os.walk(modelpath):
            for filename in filenames:
                path = os.path.join(dirpath, filename)
                if path.endswith('pth.tar'):
                    if logger is None:
                        print(f'rm {path}')
                    else:
                        logger.warning(f'rm {path}')
                    os.system("rm -rf '{}'".format(path))
            break
    path = os.path.join(modelpath, modelname)
    if logger is None:
        print('saving model to {}...'.format(path))
    else:
        logger.warning('saving model to {}...'.format(path))
    try:
        torch.save(state, path)
    except Exception as e:
        logger.error(e)


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def check_doc_size(X_train, logger):
    n_sent = []
    n_words = []
    n_words_per_doc = []
    for doc in X_train:
        n_sent.append(len(doc))
        words_per_doc = 0
        for sent in doc:
            n_words.append(len(sent))
            words_per_doc += len(sent)
        n_words_per_doc.append(words_per_doc)
    logger.info('#sent in a document')
    logger.info(pd.Series(n_sent).describe(percentiles=[.25, .5, .75, .8, .85, .9, .95, .96, .98]))
    logger.info('#words in a sent')
    logger.info(pd.Series(n_words).describe(percentiles=[.25, .5, .75, .8, .85, .9, .95, .96, .98]))
    logger.info('#words in a document')
    logger.info(pd.Series(n_words_per_doc).describe(percentiles=[.25, .5, .75, .8, .85, .9, .95, .96, .98]))


def pad_batch(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = min(np.max([len(x) for x in mini_batch]), 10)
    max_token_len = min(np.max([len(val) for sublist in mini_batch for val in sublist]), 50)
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype=np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i, j, k] = mini_batch[i][j][k]
                except IndexError:
                    pass
    return Variable(torch.from_numpy(main_matrix).transpose(0, 1))


def pad_batch_nosent_fast(args, word_index, mini_batch, region, stride):
    mini_batch_size = len(mini_batch)
    n_tokens = min(args.max_tokens, max([sum([len(sent) for sent in doc]) for doc in mini_batch]))
    main_matrix = np.zeros((mini_batch_size, n_tokens, region), dtype=np.int)
    unk_idx = word_index['UNK']
    main_matrix.fill(unk_idx)
    for i in range(mini_batch_size):
        sent_cat = [unk_idx] * (region - 1) + [word for sent in mini_batch[i] for word in sent]  # padded
        # sent_cat = [word for sent in mini_batch[i] for word in sent]
        idx = 0
        ct = 0
        last_set = set()
        while ct < n_tokens and idx < len(sent_cat):
            word_set = set()  # words in current region
            for region_idx, word in enumerate(sent_cat[idx: idx + region]):
                if word in word_set:
                    main_matrix[i][ct][region_idx] = unk_idx
                    continue
                if word != unk_idx:
                    word_set.add(word)
                main_matrix[i][ct][region_idx] = word
            if last_set == word_set:
                ct -= 1
            last_set = word_set
            idx += stride
            ct += 1

    return main_matrix


# region is for bow-cnn. need to covert vectors to multi-hot
def pad_batch_nosent(mini_batch, word_index, onehot=False, region=None, stride=None):
    mini_batch_size = len(mini_batch)
    n_tokens = min(256, max([sum([len(sent) for sent in doc]) for doc in mini_batch]))
    if onehot:
        main_matrix = np.zeros((mini_batch_size, n_tokens, 30000), dtype=np.float32)
        unk_idx = word_index['UNK']
        for i in range(mini_batch_size):
            if not region:
                ct = 0
                for sent in mini_batch[i]:
                    for word in sent:
                        if word != unk_idx:
                            if word > unk_idx:
                                word -= 1
                            main_matrix[i][ct][word] = 1
                        ct += 1
                        if ct == n_tokens:
                            break
                    if ct == n_tokens:
                        break
            else:
                sent_cat = [unk_idx] * (region - 1) + [word for sent in mini_batch[i] for word in sent]
                idx = 0
                ct = 0
                last_set = set()
                while ct < n_tokens and idx < len(sent_cat):
                    word_set = set()
                    for word in sent_cat[idx: idx + region]:
                        if word != unk_idx:
                            if word > unk_idx:
                                word -= 1
                            word_set.add(word)
                            main_matrix[i][ct][word] = 1
                    # variable-stride
                    if last_set == word_set:
                        ct -= 1
                    last_set = word_set
                    idx += stride
                    ct += 1
    else:
        main_matrix = np.zeros((mini_batch_size, n_tokens), dtype=np.int)
        for i in range(mini_batch_size):
            ct = 0
            for sent in mini_batch[i]:
                for word in sent:
                    main_matrix[i][ct] = word
                    ct += 1
                    if ct == n_tokens:
                        break
                if ct == n_tokens:
                    break
    return Variable(torch.from_numpy(main_matrix))


def iterate_minibatches(args, inputs, targets, batchsize, shuffle):
    assert inputs.shape[0] == targets.shape[0]
    if args.debug:
        for _ in range(300):
            yield inputs[:batchsize], targets[:batchsize]
        return
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
    if start_idx + batchsize < inputs.shape[0]:
        if shuffle:
            excerpt = indices[start_idx + batchsize:]
        else:
            excerpt = slice(start_idx + batchsize, start_idx + batchsize * 2)
        yield inputs[excerpt], targets[excerpt]


def iterate_minibatches_order(args, inputs, targets, batchsize):
    assert inputs.shape[0] == targets.shape[0]
    if args.debug:
        for _ in range(300):
            yield inputs[:batchsize], targets[:batchsize]
        return
    indices = np.argsort([-len(doc) for doc in inputs])
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]
    if start_idx + batchsize < inputs.shape[0]:
        excerpt = indices[start_idx + batchsize:]
        yield inputs[excerpt], targets[excerpt]


def gen_minibatch(logger, args, word_index, tokens, labels, mini_batch_size, shuffle=False):
    logger.info('# batches = {}'.format(len(tokens) / mini_batch_size))
    # for token, label in iterate_minibatches(tokens, labels, mini_batch_size, shuffle=shuffle):
    for token, label in iterate_minibatches_order(args, tokens, labels, mini_batch_size):
        if args.base_model == 'textcnn':
            token = pad_batch_nosent(token, word_index)
        elif args.base_model == 'ohcnn-seq':
            token = pad_batch_nosent(token, word_index, onehot=True)
        elif args.base_model == 'ohcnn-bow':
            token = pad_batch_nosent(token, word_index, onehot=True, region=20, stride=2)
        elif args.base_model == 'ohcnn-bow-fast':
            main_matrix = pad_batch_nosent_fast(args, word_index, token, region=20, stride=2)
            token = Variable(torch.from_numpy(main_matrix))
        else:
            token = pad_batch(token)
        if args.gpu:
            yield token.cuda(), label
        else:
            yield token, label


def gen_minibatch_from_cache(logger, args, tree, mini_batch_size, name, shuffle):
    pkl_path = '{}_{}.pkl'.format(name, mini_batch_size)
    if not os.path.exists(pkl_path):
        logger.error('{} NOT FOUND'.format(pkl_path))
        exit(-1)
    if 'train' in name:
        if tree.data_cache is not None:
            (token_l, label_l) = tree.data_cache
            logger.info('loaded from tree.data_cache')
        else:
            (token_l, label_l) = pickle.load(open(pkl_path, 'rb'))
            tree.data_cache = (token_l, label_l)
    else:
        (token_l, label_l) = pickle.load(open(pkl_path, 'rb'))
    logger.info('loaded {} batches from {}'.format(len(label_l), pkl_path))
    if args.debug:
        for _ in range(1000):
            token = Variable(torch.from_numpy(token_l[0]))
            label = label_l[0]
            if args.gpu:
                yield token.cuda(), label
            else:
                yield token, label
        return
    if shuffle:
        indices = np.arange(len(token_l))
        np.random.shuffle(indices)
        for i in indices:
            token = token_l[i]
            label = label_l[i]
            token = Variable(torch.from_numpy(token))
            if args.gpu:
                yield token.cuda(), label
            else:
                yield token, label
    else:
        for token, label in zip(token_l, label_l):
            # out of memory
            if mini_batch_size > 32:
                new_batch_size = mini_batch_size // 2
                for i in range(0, mini_batch_size, new_batch_size):
                    token_v = Variable(torch.from_numpy(token[i:i + new_batch_size]))
                    label_v = label[i:i + new_batch_size]
                    if args.gpu:
                        yield token_v.cuda(), label_v
                    else:
                        yield token_v, label_v
            else:
                token = Variable(torch.from_numpy(token))
                if args.gpu:
                    yield token.cuda(), label
                else:
                    yield token, label


def save_minibatch(logger, args, word_index, tokens, labels, mini_batch_size, name=''):
    filename = '{}_{}.pkl'.format(name, mini_batch_size)
    if os.path.exists(filename):
        logger.warning(f'skipped since {filename} existed')
        return
    token_l = []
    label_l = []
    for token, label in tqdm(iterate_minibatches_order(args, tokens, labels, mini_batch_size)):
        token = pad_batch_nosent_fast(args, word_index, token, region=20, stride=2)
        token_l.append(token)
        label_l.append(label)
    pickle.dump((token_l, label_l), open(filename, 'wb'))
