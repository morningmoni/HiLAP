from collections import defaultdict
from tqdm import tqdm


def read_rcv1_ids(filepath):
    ids = set()
    with open(filepath) as f:
        new_doc = True
        for line in f:
            line_split = line.strip().split()
            if new_doc and len(line_split) == 2:
                tmp, did = line_split
                if tmp == '.I':
                    ids.add(did)
                    new_doc = False
                else:
                    print(line_split)
                    print('maybe error')
            elif len(line_split) == 0:
                new_doc = True
    print('{} samples in {}'.format(len(ids), filepath))
    return ids


def read_rcv1():
    p2c = defaultdict(list)
    id2doc = defaultdict(lambda: defaultdict(list))
    nodes = defaultdict(lambda: defaultdict(list))
    with open('rcv1/rcv1.topics.hier.orig.txt') as f:
        for line in f:
            start = line.find('parent: ') + len('parent: ')
            end = line.find(' ', start)
            parent = line[start:end]
            start = line.find('child: ') + len('child: ')
            end = line.find(' ', start)
            child = line[start:end]
            start = line.find('child-description: ') + len('child-description: ')
            end = line.find('\n', start)
            child_desc = line[start:end]
            p2c[parent].append(child)
    for label in p2c:
        if label == 'None':
            continue
        for children in p2c[label]:
            nodes[label]['children'].append(children)
            nodes[children]['parent'].append(label)

    with open('rcv1/rcv1-v2.topics.qrels') as f:
        for line in f:
            cat, doc_id, _ = line.strip().split()
            id2doc[doc_id]['categories'].append(cat)
    X_train = []
    X_test = []
    train_ids = []
    test_ids = []
    train_id_set = read_rcv1_ids('../datasets/rcv1_token/lyrl2004_tokens_train.dat')
    test_id_set = read_rcv1_ids('../datasets/rcv1_token/lyrl2004_tokens_test_pt0.dat')
    test_id_set |= read_rcv1_ids('../datasets/rcv1_token/lyrl2004_tokens_test_pt1.dat')
    test_id_set |= read_rcv1_ids('../datasets/rcv1_token/lyrl2004_tokens_test_pt2.dat')
    test_id_set |= read_rcv1_ids('../datasets/rcv1_token/lyrl2004_tokens_test_pt3.dat')
    print('len(test) total={}'.format(len(test_id_set)))
    n_not_found = 0
    with open('rcv1/docs.txt') as f:
        for line in tqdm(f):
            doc_id, text = line.strip().split(maxsplit=1)
            if doc_id in train_id_set:
                train_ids.append(doc_id)
                X_train.append(text)
            elif doc_id in test_id_set:
                test_ids.append(doc_id)
                X_test.append(text)
            else:
                n_not_found += 1
    print('there are {} that cannot be found in official tokenized rcv1'.format(n_not_found))
    print('len(train_ids)={} len(test_ids)={}'.format(len(train_ids), len(test_ids)))
    return X_train, X_test, train_ids, test_ids, dict(id2doc), dict(nodes)


if __name__ == '__main__':
    read_rcv1()
