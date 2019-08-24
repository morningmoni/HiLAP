import numpy as np
import torch
from sklearn.datasets import fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

from readData_fungo import read_fungo
from readData_nyt import read_nyt
from readData_yelp import read_yelp


def rcv1_test(rcv1):
    X_train = rcv1.data[:23149]
    Y_train = rcv1.target[:23149]
    X_test = rcv1.data[23149:]
    Y_test = rcv1.target[23149:]
    return X_train, Y_train, X_test, Y_test


def yelp_test():
    subtree_name = 'root'
    X_train, X_test, train_ids, test_ids, business_dict, nodes = read_yelp(subtree_name, 5, 10)
    print(f'#training={len(train_ids)} #test={len(test_ids)}')
    n_tokens = 256
    print(f'use only first {n_tokens} tokens')
    X_train = [' '.join(i.split()[:n_tokens]) for i in X_train]
    X_test = [' '.join(i.split()[:n_tokens]) for i in X_test]
    print('fit_transform...')
    tf = TfidfVectorizer()
    X_train = tf.fit_transform(X_train)
    X_test = tf.transform(X_test)
    Y_train = [business_dict[bid]['categories'] for bid in train_ids]
    Y_test = [business_dict[bid]['categories'] for bid in test_ids]
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)
    Y_test = mlb.transform(Y_test)
    return X_train, Y_train, X_test, Y_test, train_ids, test_ids


def nyt_test():
    X_train, X_test, train_ids, test_ids, id2doc, nodes = read_nyt()
    print(f'#training={len(train_ids)} #test={len(test_ids)}')
    n_tokens = 256
    print(f'use only first {n_tokens} tokens')
    X_train = [' '.join(i.split()[:n_tokens]) for i in X_train]
    X_test = [' '.join(i.split()[:n_tokens]) for i in X_test]
    print('fit_transform...')
    tf = TfidfVectorizer()
    X_train = tf.fit_transform(X_train)
    X_test = tf.transform(X_test)
    Y_train = [id2doc[bid]['categories'] for bid in train_ids]
    Y_test = [id2doc[bid]['categories'] for bid in test_ids]
    mlb = MultiLabelBinarizer()
    Y_train = mlb.fit_transform(Y_train)
    Y_test = mlb.transform(Y_test)
    return X_train, Y_train, X_test, Y_test, train_ids, test_ids


def fungo_test(data_name):
    X_train, X_test, train_ids, test_ids, id2doc, nodes = read_fungo(data_name)
    Y_train = [id2doc[bid]['categories'] for bid in train_ids]
    Y_test = [id2doc[bid]['categories'] for bid in test_ids]
    # Actually here Y is not used. We use id2doc for labels.
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(np.concatenate([Y_train, Y_test]))
    Y_train = Y[:len(Y_train)]
    Y_test = Y[-len(Y_test):]

    return X_train, Y_train, X_test, Y_test, train_ids, test_ids


def my_collate(batch):
    features = torch.FloatTensor([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return [features, labels]


class featureDataset(Dataset):
    def __init__(self, data_name, train=True):
        self.train = train
        self.data = data_name
        if data_name == 'rcv1':
            self.rcv1 = fetch_rcv1()
            X_train, Y_train, X_test, Y_test = rcv1_test(self.rcv1)
            if train:
                self.samples = X_train
            else:
                self.samples = X_test
        else:
            if data_name == 'yelp':
                X_train, Y_train, X_test, Y_test, train_ids, test_ids = yelp_test()
            elif data_name == 'nyt':
                X_train, Y_train, X_test, Y_test, train_ids, test_ids = nyt_test()
            else:
                X_train, Y_train, X_test, Y_test, train_ids, test_ids = fungo_test(data_name)
            if train:
                self.samples = X_train
                self.ids = train_ids
            else:
                self.samples = X_test
                self.ids = test_ids

    def __len__(self):
        if 'FUN' in self.data or 'GO' in self.data:
            return len(self.samples)
        if self.data == 'fungo':
            return len(self.samples)
        return self.samples.shape[0]

    def __getitem__(self, item):
        if self.data == 'rcv1':
            if self.train:
                vector, label = self.samples[item].todense().tolist()[0], str(int(self.rcv1.sample_id[item]))
            else:
                vector, label = self.samples[item].todense().tolist()[0], str(int(self.rcv1.sample_id[item + 23149]))
        elif self.data in ['yelp', 'nyt']:
            vector, label = self.samples[item].todense().tolist()[0], self.ids[item].strip().strip('\n')
        elif 'FUN' in self.data or 'GO' in self.data or self.data == 'fungo':
            vector, label = self.samples[item], self.ids[item]
        return vector, label
