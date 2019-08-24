import os
import pickle
import random
from collections import defaultdict

import numpy as np
from IPython import embed


def replace_nan_by_mean(X):
    # Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = np.nanmean(X, axis=0)
    # Find indicies that you need to replace
    inds = np.where(np.isnan(X))
    # Place column means in the indices. Align the arrays using take
    X[inds] = np.take(col_mean, inds[1])
    return X


def read_fungo_(f_in):
    """
    X: features
    Y: labels
    C: for *_FUN: in format ancestor1/ancestor2/.../label
       for *_GO: in format parent/child [CAUTION: since it is a DAG, its size is larger than C_slash]
    C_slash: unique labels for *_GO [somehow have 3 more labels than HMCN's paper.]

    """
    ct = 0
    A = 0
    C = 0
    C_set = set()
    flag = False
    X = []
    Y = []
    with open(f_in) as f:
        for line in f:
            if line.startswith('@ATTRIBUTE'):
                if '/' in line:
                    # print(line.split(','))
                    C = line.strip().split(',')
                    C[0] = C[0].split()[-1]
                    # print([i.split('/')[-1] for i in C][-10:])
                    C_slash = set([i.split('/')[-1] for i in C])
                else:
                    A += 1
            if flag:
                ct += 1
                data = line.strip().split(',')
                classes = data[-1].split('@')
                # convert ? to nan
                X.append([float(i) if i != '?' else np.nan for i in data[:-1]])
                Y.append(classes)
                C_set.update(classes)
            if line.startswith('@DATA'):
                flag = True
    X = np.array(X)
    X = replace_nan_by_mean(X)
    # print(f'[{f_in}] #features={A}, #Classes={len(C)}, #C_slash={len(C_slash)}, #C_appear={len(C_set)}, #samples={ct}')
    return X, Y, C, C_slash


def read_fungo_all(name):
    valid = read_fungo_(
        f"./protein_datasets/{name}.valid.arff")
    test = read_fungo_(
        f"./protein_datasets/{name}.test.arff")
    train = read_fungo_(
        f"./protein_datasets/{name}.train.arff")
    return train, valid, test


def construct_hier_dag(name):
    hierarchy = defaultdict(set)
    train_classes = read_fungo_(
        f"./protein_datasets/{name}.train.arff")[2]
    valid_classes = read_fungo_(
        f"./protein_datasets/{name}.valid.arff")[2]
    test_classes = read_fungo_(
        f"./protein_datasets/{name}.test.arff")[2]
    for t in [train_classes, valid_classes, test_classes]:
        for y in t:
            parent, child = y.split('/')
            if parent == 'root':
                parent = 'Top'
            hierarchy[parent].add(child)
            if not child in hierarchy:
                hierarchy[child] = set()
    return hierarchy


def construct_hierarchy(name):
    # Construct hierarchy from train/valid/test.
    hierarchy = defaultdict(set)
    train_classes = read_fungo_(
        f"./protein_datasets/{name}.train.arff")[2]
    valid_classes = read_fungo_(
        f"./protein_datasets/{name}.valid.arff")[2]
    test_classes = read_fungo_(
        f"./protein_datasets/{name}.test.arff")[2]
    for t in [train_classes, valid_classes, test_classes]:
        for y in t:
            hier_list = y.split('/')
            # Add a pseudo node: Top
            if len(hier_list) == 1:
                hierarchy['Top'].add(hier_list[0])
                hierarchy[hier_list[0]] = set()
                continue
            for l in range(1, len(hier_list)):
                parent = '/'.join(hier_list[:l])
                child = '/'.join(hier_list[:l + 1])
                if l == 1:
                    hierarchy['Top'].add(parent)
                if not child in hierarchy[parent]:
                    hierarchy[parent].add(child)
                if not child in hierarchy:
                    hierarchy[child] = set()
    return hierarchy


def get_all_ancestor_nodes(hierarchy, node):
    node_list = set()

    def dfs(node):
        if node != 'Top':
            node_list.add(node)
        parents = hierarchy[node]['parent']
        for parent in parents:
            dfs(parent)

    dfs(node)
    return node_list


def read_go(name):
    if os.path.exists(f'./protein_datasets/{name}.pkl'):
        return pickle.load(open(f'./protein_datasets/{name}.pkl', 'rb'))
    p2c = defaultdict(list)
    id2doc = defaultdict(lambda: defaultdict(list))
    nodes = defaultdict(lambda: defaultdict(list))
    random.seed(42)

    train, valid, test = read_fungo_all(name)
    hierarchy = construct_hier_dag(name)
    for parent in hierarchy:
        for child in hierarchy[parent]:
            p2c[parent].append(child)
    for label in p2c:
        for children in p2c[label]:
            nodes[label]['children'].append(children)
            nodes[children]['parent'].append(label)

    train_data = np.concatenate([train[0], valid[0]])
    train[1].extend(valid[1])

    X_train = []
    X_test = []
    train_ids = []
    test_ids = []

    for idx, (feature, classes) in enumerate(zip(train_data, train[1])):
        X_train.append(feature)
        train_ids.append(idx)
        for class_ in classes:
            ancestor_nodes = get_all_ancestor_nodes(nodes, class_)
            for label in ancestor_nodes:
                if not label in id2doc[idx]['categories']:
                    id2doc[idx]['categories'].append(label)

    for idx, (feature, classes) in enumerate(zip(test[0], test[1])):
        X_test.append(feature)
        test_ids.append(idx + train_data.shape[0])
        for class_ in classes:
            ancestor_nodes = get_all_ancestor_nodes(nodes, class_)
            for label in ancestor_nodes:
                if not label in id2doc[idx + train_data.shape[0]]['categories']:
                    id2doc[idx + train_data.shape[0]]['categories'].append(label)
    res = X_train, X_test, train_ids, test_ids, dict(id2doc), dict(nodes)
    pickle.dump(res, open(f'./protein_datasets/{name}.pkl', 'wb'))
    return res


def read_fun(name):
    if os.path.exists(f'./protein_datasets/{name}.pkl'):
        return pickle.load(open(f'./protein_datasets/{name}.pkl', 'rb'))
    p2c = defaultdict(list)
    id2doc = defaultdict(lambda: defaultdict(list))
    nodes = defaultdict(lambda: defaultdict(list))
    random.seed(42)

    train, valid, test = read_fungo_all(name)
    hierarchy = construct_hierarchy(name)
    for parent in hierarchy:
        for child in hierarchy[parent]:
            p2c[parent].append(child)
    for label in p2c:
        for children in p2c[label]:
            nodes[label]['children'].append(children)
            nodes[children]['parent'].append(label)

    train_data = np.concatenate([train[0], valid[0]])
    train[1].extend(valid[1])

    X_train = []
    X_test = []
    train_ids = []
    test_ids = []

    for idx, (feature, classes) in enumerate(zip(train_data, train[1])):
        X_train.append(feature)
        train_ids.append(idx)
        for class_ in classes:
            hier_list = class_.split('/')
            for l in range(1, len(hier_list) + 1):
                label = '/'.join(hier_list[:l])
                if not label in id2doc[idx]['categories']:
                    id2doc[idx]['categories'].append(label)

    for idx, (feature, classes) in enumerate(zip(test[0], test[1])):
        X_test.append(feature)
        test_ids.append(idx + train_data.shape[0])
        for class_ in classes:
            # For each sample, we treat all nodes in the path as labels.
            hier_list = class_.split('/')
            for l in range(1, len(hier_list) + 1):
                label = '/'.join(hier_list[:l])
                if not label in id2doc[idx + train_data.shape[0]]['categories']:
                    id2doc[idx + train_data.shape[0]]['categories'].append(label)
    res = X_train, X_test, train_ids, test_ids, dict(id2doc), dict(nodes)
    pickle.dump(res, open(f'./protein_datasets/{name}.pkl', 'wb'))
    return res


def read_fungo(name):
    if 'FUN' in name:
        return read_fun(name)
    return read_go(name)


def process_for_cssag(name, train_data, train_labels, test_data, test_labels):
    if 'FUN' in name:
        hierarchy = construct_hierarchy(name)
    else:
        hierarchy = construct_hier_dag(name)
    nodes = defaultdict(lambda: defaultdict(list))
    p2c = defaultdict(list)
    for parent in hierarchy:
        for child in hierarchy[parent]:
            p2c[parent].append(child)
    for label in p2c:
        for children in p2c[label]:
            nodes[label]['children'].append(children)
            nodes[children]['parent'].append(label)
    label2id = {}
    for k in hierarchy:
        if k not in label2id:
            label2id[k] = len(label2id)
    with open('./cssag/' + name + '.hier', 'w') as OUT:
        for k in hierarchy:
            OUT.write(str(label2id[k]))
            for c in hierarchy[k]:
                OUT.write(' ' + str(label2id[c]))
            OUT.write('\n')
    with open('./cssag/' + name + '.train.x', 'w') as OUT:
        for l in range(train_data.shape[0]):
            x = train_data[l]
            OUT.write('0')
            for idx, i in enumerate(x):
                OUT.write(' ' + str(idx + 1) + ':' + str(i))
            OUT.write('\n')
    with open('./cssag/' + name + '.test.x', 'w') as OUT:
        for l in range(test_data.shape[0]):
            x = test_data[l]
            OUT.write('0')
            for idx, i in enumerate(x):
                OUT.write(' ' + str(idx + 1) + ':' + str(i))
            OUT.write('\n')
    if 'FUN' in name:
        with open('./cssag/' + name + '.train.y', 'w') as OUT:
            for classes in train_labels:
                # OUT.write(str(label2id['Top']))
                labels = set()
                for class_ in classes:
                    hier_list = class_.split('/')
                    for l in range(1, len(hier_list) + 1):
                        label = '/'.join(hier_list[:l])
                        labels.add(label)
                for i, label in enumerate(labels):
                    if i > 0:
                        OUT.write(',')
                    OUT.write(str(label2id[label]))
                OUT.write('\n')
        with open('./cssag/' + name + '.test.y', 'w') as OUT:
            for classes in test_labels:
                # OUT.write(str(label2id['Top']))
                labels = set()
                for class_ in classes:
                    hier_list = class_.split('/')
                    for l in range(1, len(hier_list) + 1):
                        label = '/'.join(hier_list[:l])
                        labels.add(label)
                for i, label in enumerate(labels):
                    if i > 0:
                        OUT.write(',')
                    OUT.write(str(label2id[label]))
                OUT.write('\n')

    elif 'GO' in name:
        with open('./cssag/' + name + '.train.y', 'w') as OUT:
            for classes in train_labels:
                labels = set()
                for class_ in classes:
                    ancestor_nodes = get_all_ancestor_nodes(nodes, class_)
                    for label in ancestor_nodes:
                        labels.add(label)
                for i, label in enumerate(labels):
                    if i > 0:
                        OUT.write(',')
                    OUT.write(str(label2id[label]))
                OUT.write('\n')
        with open('./cssag/' + name + '.test.y', 'w') as OUT:
            for classes in test_labels:
                labels = set()
                for class_ in classes:
                    ancestor_nodes = get_all_ancestor_nodes(nodes, class_)
                    for label in ancestor_nodes:
                        labels.add(label)
                for i, label in enumerate(labels):
                    if i > 0:
                        OUT.write(',')
                    OUT.write(str(label2id[label]))
                OUT.write('\n')


if __name__ == '__main__':
    # res = read_go()
    # embed()
    # exit()
    name = 'eisen_FUN'
    train, valid, test = read_fungo_all(name)
    train_data = np.concatenate([train[0], valid[0]])
    train[1].extend(valid[1])
    train_labels = train[1]
    process_for_cssag(name, train_data, train_labels, test[0], test[1])
    embed()
    exit()
