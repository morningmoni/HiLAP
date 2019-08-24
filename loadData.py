import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from Logger_morning import myLogger
from readData_rcv1 import read_rcv1
from readData_yelp import read_yelp
from readData_nyt import read_nyt

spacy_en = spacy.load('en')
logger = myLogger('exp')


def tokenizer(text):  # create a tokenizer function
    # return text.lower().split()
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]


def read_word_embed(pre_trained_path, biovec=False):
    logger.info('loading pre-trained embedding from {}'.format(pre_trained_path))
    if not biovec:
        with open(pre_trained_path) as f:
            words, vectors = zip(*[line.strip().split(' ', 1) for line in f])
            wv = np.loadtxt(vectors)
    else:
        with open(pre_trained_path + 'types.txt') as f:
            words = [line.strip() for line in f]
        wv = np.loadtxt(pre_trained_path + 'vectors.txt')
    return words, wv


def prepare_word(pre_trained_path, vocab, biovec):
    words, wv = read_word_embed(pre_trained_path, biovec)
    unknown_vector = np.random.random_sample((wv.shape[1],))
    word_set = set(words)
    unknown_words = list(set(vocab).difference(set(words)))
    logger.info('there are {} OOV words'.format(len(unknown_words)))
    word_index = {w: i for i, w in enumerate(words)}
    unknown_word_vectors = [np.add.reduce([wv[word_index[w]] if w in word_set else unknown_vector
                                           for w in word.split(' ')])
                            for word in unknown_words]
    wv = np.vstack((wv, unknown_word_vectors))
    words = list(words) + unknown_words
    # Normalize each row (word vector) in the matrix to sum-up to 1
    row_norm = np.sum(np.abs(wv) ** 2, axis=-1) ** (1. / 2)
    wv /= row_norm[:, np.newaxis]

    word_index = {w: i for i, w in enumerate(words)}
    return wv, word_index


def prepare_word_filter(pre_trained_path, vocab, biovec, shared_unk=False, norm=True):
    words, wv = read_word_embed(pre_trained_path, biovec)
    known_words = set(vocab) & set(words)
    unknown_words = set(vocab).difference(set(words))
    logger.info('there are {} OOV words'.format(len(unknown_words)))

    word_index = {w: i for i, w in enumerate(words)}
    new_word_index = {}

    filter_idx = []
    if shared_unk:
        ct = 1
    else:
        ct = 0
    for word in known_words:
        new_word_index[word] = ct
        filter_idx.append(word_index[word])
        ct += 1
    for word in unknown_words:
        if shared_unk:
            new_word_index[word] = 0
        else:
            new_word_index[word] = ct
            ct += 1

    wv = wv[filter_idx]
    if shared_unk:
        unknown_vector = np.random.random_sample((wv.shape[1],)) - .5
        wv = np.vstack((unknown_vector, wv))
    else:
        unknown_vectors = np.random.random_sample((len(unknown_words), wv.shape[1])) - .5
        wv = np.vstack((wv, unknown_vectors))

    # Normalize each row (word vector) in the matrix to sum-up to 1
    if norm:
        row_norm = np.sum(np.abs(wv) ** 2, axis=-1) ** (1. / 2)
        wv /= row_norm[:, np.newaxis]

    return wv, new_word_index


def tokenize_text_normal(X_l):
    X = []
    vocab = set()
    for text in tqdm(X_l):
        X.append([])
        for sent in sent_tokenize(text):
            t = tokenizer(sent)
            vocab.update(t)
            X[-1].append(t)
    return X, vocab


def oov2unk(X_train, X_test, vocab_train, vocab_test, vocab_size=None, word_min_sup=5, count_test=False, filter=False):
    if not vocab_size:
        logger.info('word_min_sup={}'.format(word_min_sup))
    vocab = vocab_train | vocab_test
    logger.info(
        "[before]vocab_train:{} vocab_test:{} vocab:{}".format(len(vocab_train), len(vocab_test), len(vocab)))
    word_count = defaultdict(int)
    if filter:
        word_w_num = set()
        with open('rcv1/rcv1_stopword.txt') as f:
            stop_word = set([line.strip() for line in f])
    for doc in X_train:
        for sent in doc:
            for word in sent:
                if filter:
                    if word in stop_word:
                        continue
                    if any(c.isdigit() for c in word):
                        word_w_num.add(word)
                        continue
                word_count[word] += 1
    if filter:
        logger.info('removeNumbers {} {}'.format(len(word_w_num), word_w_num))
    if count_test:
        for doc in X_test:
            for sent in doc:
                for word in sent:
                    word_count[word] += 1
    if vocab_size:
        logger.info(f'limit vocab_size={vocab_size}')
        vocab_by_freq = set([k for k in sorted(word_count, key=word_count.get, reverse=True)][:vocab_size])
        X_train = [[[word if word in vocab_by_freq else 'UNK' for word in sent] for sent in doc] for doc in
                   X_train]
        X_test = [[[word if word in vocab_by_freq else 'UNK' for word in sent] for sent in doc] for doc in
                  X_test]
    else:
        X_train = [[[word if word_count[word] >= word_min_sup else 'UNK' for word in sent] for sent in doc] for doc in
                   X_train]
        X_test = [[[word if word_count[word] >= word_min_sup else 'UNK' for word in sent] for sent in doc] for doc in
                  X_test]
    logger.info('Tokenization may be poor. Next are some examples:')
    logger.info(X_train[:5])
    vocab_train = set([word for doc in X_train for sent in doc for word in sent])
    vocab_test = set([word for doc in X_test for sent in doc for word in sent])
    vocab = vocab_train | vocab_test
    logger.info(
        "[after]vocab_train:{} vocab_test:{} vocab:{}".format(len(vocab_train), len(vocab_test), len(vocab)))
    return X_train, X_test, vocab


def load_data_rcv1_onehot(suffix):
    if os.path.exists('preload_data{}_onehot.pkl'.format(suffix)):
        logger.warn('loading from preload_data{}_onehot.pkl'.format(suffix))
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = pickle.load(
            open('preload_data{}_onehot.pkl'.format(suffix), 'rb'))
        return X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes

    _, _, train_ids, test_ids, id2doc, wv, word_index, nodes = pickle.load(
        open('preload_data{}.pkl'.format(suffix), 'rb'))
    X_train, vocab_train, X_test, vocab_test = pickle.load(open('text_tokenized{}.pkl'.format(suffix), 'rb'))
    X_train, X_test, vocab = oov2unk(X_train, X_test, vocab_train, vocab_test, vocab_size=30000, filter=True)
    word_index = {w: ct for ct, w in enumerate(vocab)}
    X_train = [[[word_index[word] for word in sent] for sent in doc] for doc in X_train]
    X_test = [[[word_index[word] for word in sent] for sent in doc] for doc in X_test]
    logger.info('saving preload_data{}_onehot.pkl'.format(suffix))
    res = np.array(X_train), np.array(X_test), np.array(train_ids), np.array(test_ids), id2doc, wv, word_index, nodes
    pickle.dump(res, open('preload_data{}_onehot.pkl'.format(suffix), 'wb'))
    return res


def load_data_rcv1(embedding_path, suffix):
    if os.path.exists('preload_data{}.pkl'.format(suffix)):
        logger.warn('loading from preload_data{}.pkl'.format(suffix))
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = pickle.load(
            open('preload_data{}.pkl'.format(suffix), 'rb'))
        return X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes
    X_train, X_test, train_ids, test_ids, id2doc, nodes = read_rcv1()
    if os.path.exists('text_tokenized{}.pkl'.format(suffix)):
        X_train, vocab_train, X_test, vocab_test = pickle.load(open('text_tokenized{}.pkl'.format(suffix), 'rb'))
    else:
        X_train, vocab_train = tokenize_text_normal(X_train)
        X_test, vocab_test = tokenize_text_normal(X_test)
        res = X_train, vocab_train, X_test, vocab_test
        pickle.dump(res, open('text_tokenized{}.pkl'.format(suffix), 'wb'))
    X_train, X_test, vocab = oov2unk(X_train, X_test, vocab_train, vocab_test, count_test=True)
    wv, word_index = prepare_word_filter(embedding_path, vocab, biovec=False)
    X_train = [[[word_index[word] for word in sent] for sent in doc] for doc in X_train]
    X_test = [[[word_index[word] for word in sent] for sent in doc] for doc in X_test]
    logger.info('saving preload_data{}.pkl'.format(suffix))
    res = np.array(X_train), np.array(X_test), np.array(train_ids), np.array(test_ids), id2doc, wv, word_index, nodes
    pickle.dump(res, open('preload_data{}.pkl'.format(suffix), 'wb'))
    return res


def load_data_nyt_onehot(suffix):
    if os.path.exists('preload_data{}_onehot.pkl'.format(suffix)):
        logger.warn('loading from preload_data{}_onehot.pkl'.format(suffix))
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = pickle.load(
            open('preload_data{}_onehot.pkl'.format(suffix), 'rb'))
        return X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes

    _, _, train_ids, test_ids, id2doc, wv, word_index, nodes = pickle.load(
        open('preload_data{}.pkl'.format(suffix), 'rb'))
    X_train, vocab_train, X_test, vocab_test = pickle.load(open('text_tokenized{}.pkl'.format(suffix), 'rb'))
    X_train, X_test, vocab = oov2unk(X_train, X_test, vocab_train, vocab_test, vocab_size=30000, filter=True)
    word_index = {w: ct for ct, w in enumerate(vocab)}
    X_train = [[[word_index[word] for word in sent] for sent in doc] for doc in X_train]
    X_test = [[[word_index[word] for word in sent] for sent in doc] for doc in X_test]
    logger.info('saving preload_data{}_onehot.pkl'.format(suffix))
    res = np.array(X_train), np.array(X_test), np.array(train_ids), np.array(test_ids), id2doc, wv, word_index, nodes
    pickle.dump(res, open('preload_data{}_onehot.pkl'.format(suffix), 'wb'))
    return res


def load_data_nyt(embedding_path, suffix):
    if os.path.exists('preload_data{}.pkl'.format(suffix)):
        logger.warn('loading from preload_data{}.pkl'.format(suffix))
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = pickle.load(
            open('preload_data{}.pkl'.format(suffix), 'rb'))
        return X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes
    X_train, X_test, train_ids, test_ids, id2doc, nodes = read_nyt()
    if os.path.exists('text_tokenized{}.pkl'.format(suffix)):
        X_train, vocab_train, X_test, vocab_test = pickle.load(open('text_tokenized{}.pkl'.format(suffix), 'rb'))
    else:
        X_train, vocab_train = tokenize_text_normal(X_train)
        X_test, vocab_test = tokenize_text_normal(X_test)
        res = X_train, vocab_train, X_test, vocab_test
        pickle.dump(res, open('text_tokenized{}.pkl'.format(suffix), 'wb'))
    X_train, X_test, vocab = oov2unk(X_train, X_test, vocab_train, vocab_test, count_test=True)
    wv, word_index = prepare_word_filter(embedding_path, vocab, biovec=False)
    X_train = [[[word_index[word] for word in sent] for sent in doc] for doc in X_train]
    X_test = [[[word_index[word] for word in sent] for sent in doc] for doc in X_test]
    logger.info('saving preload_data{}.pkl'.format(suffix))
    res = np.array(X_train), np.array(X_test), np.array(train_ids), np.array(test_ids), id2doc, wv, word_index, nodes
    pickle.dump(res, open('preload_data{}.pkl'.format(suffix), 'wb'))
    return res


def load_data_yelp(embedding_path, suffix, root, min_reviews=1, max_reviews=10):
    """
    suffix: used to distinguish different pkl files
    root: which subtree to use e.g., root (use all nodes as classes, 1004 - 1 in total), Hotels & Travel
    min_reviews: remove businesses that have < min_reviews
    max_reviews: use at most max_reviews for a business
    """
    if os.path.exists('preload_data{}.pkl'.format(suffix)):
        logger.warn('loading from preload_data{}.pkl'.format(suffix))
        X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes = pickle.load(
            open('preload_data{}.pkl'.format(suffix), 'rb'))
        return X_train, X_test, train_ids, test_ids, id2doc, wv, word_index, nodes
    logger.warn('reviews min_sup={}, max_sup={}'.format(min_reviews, max_reviews))
    X_train, X_test, train_ids, test_ids, id2doc, nodes = read_yelp(root, min_reviews, max_reviews)
    # TODO need to make sure every time train_ids, test_ids are the same
    if os.path.exists('text_tokenized{}.pkl'.format(suffix)):
        logger.warn('loading from text_tokenized{}.pkl'.format(suffix))
        X_train, vocab_train, X_test, vocab_test = pickle.load(open('text_tokenized{}.pkl'.format(suffix), 'rb'))
    else:
        X_train, vocab_train = tokenize_text_normal(X_train)
        X_test, vocab_test = tokenize_text_normal(X_test)
        res = X_train, vocab_train, X_test, vocab_test
        pickle.dump(res, open('text_tokenized{}.pkl'.format(suffix), 'wb'))
    X_train, X_test, vocab = oov2unk(X_train, X_test, vocab_train, vocab_test, vocab_size=30000)
    wv, word_index = prepare_word_filter(embedding_path, vocab, biovec=False)
    X_train = [[[word_index[word] for word in sent] for sent in doc] for doc in X_train]
    X_test = [[[word_index[word] for word in sent] for sent in doc] for doc in X_test]
    logger.info('saving preload_data{}.pkl'.format(suffix))
    res = np.array(X_train), np.array(X_test), np.array(train_ids), np.array(test_ids), id2doc, wv, word_index, nodes
    pickle.dump(res, open('preload_data{}.pkl'.format(suffix), 'wb'))
    return res


def filter_ancestors(id2doc, nodes):
    logger.info('keep only lowest label in a path...')
    id2doc_na = defaultdict(dict)
    labels_ct = []
    lowest_labels_ct = []
    for bid in id2doc:
        lowest_labels = []
        cat_set = set(id2doc[bid]['categories'])
        for label in id2doc[bid]['categories']:
            if len(set(nodes[label]['children']) & cat_set) == 0:
                lowest_labels.append(label)
        labels_ct.append(len(cat_set))
        lowest_labels_ct.append(len(lowest_labels))
        id2doc_na[bid]['categories'] = lowest_labels
    logger.info('#labels')
    logger.info(pd.Series(labels_ct).describe(percentiles=[.25, .5, .75, .8, .85, .9, .95, .96, .98]))
    logger.info('#lowest labels')
    logger.info(pd.Series(lowest_labels_ct).describe(percentiles=[.25, .5, .75, .8, .85, .9, .95, .96, .98]))
    return id2doc_na


def split_multi(X_train, train_ids, id2doc_train, id2doc):
    logger.info('split one sample with m labels to m samples...')
    X_train_new = []
    train_ids_new = []
    id2doc_train_new = defaultdict(dict)
    for X, did in zip(X_train, train_ids):
        ct = 0
        for label in id2doc_train[did]['categories']:
            newID = '{}-{}'.format(did, ct)
            id2doc[newID] = id2doc[did]
            X_train_new.append(X)
            train_ids_new.append(newID)
            id2doc_train_new[newID]['categories'] = [label]
            ct += 1
    return np.array(X_train_new), np.array(train_ids_new), id2doc_train_new, id2doc
