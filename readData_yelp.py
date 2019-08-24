import ast
import random

from tqdm import tqdm


def read_yelp_files(root, review_min_sup):
    nodes = {}
    with open('yelp/Taxonomy_100') as f:
        for line in f:
            node = ast.literal_eval(line)
            nodes[node['title']] = node
    subtree_name = root
    node_set = set()
    q = [subtree_name]
    nextq = []
    print(q)
    while len(q) > 0:
        cur_name = q.pop(0)
        node_set.add(cur_name)
        print(nodes[cur_name]['children'], end=',')
        nextq.extend(nodes[cur_name]['children'])
        if len(q) == 0:
            q = nextq
            nextq = []
            print()
    print('keep {} nodes in total'.format(len(node_set)))
    print(node_set)
    nodes = {node: nodes[node] for node in node_set}

    business_dict = {}
    ct = 0
    below_min_sup = 0
    with open('yelp/yelp_data_100.csv') as f:
        for line in tqdm(f):
            business = ast.literal_eval(line)
            if len(business['reviews']) < review_min_sup:
                below_min_sup += 1
                continue
            labels = set(business['categories']) & node_set
            if len(labels) == 0:
                continue
            if len(labels) != len(business['categories']):
                ct += 1
                # print('there are labels on other subtree: {} {}'.format(labels, business['categories']))
            business['categories'] = labels
            business_dict[business['business_id']] = business
    print('keep {} business for tree {}'.format(len(business_dict), subtree_name))
    print('there are {} that have labels on other subtrees'.format(ct))
    print('there are {} that are filtered because of min_sup'.format(below_min_sup))
    return nodes, business_dict


def read_yelp(root, min_reviews=1, max_reviews=10):
    nodes, business_dict = read_yelp_files(root, min_reviews)
    random.seed(42)
    train_ids = []
    test_ids = []
    X_train = []
    X_test = []

    for bid in business_dict:
        reviews_concat = ' '.join(business_dict[bid]['reviews'][:max_reviews])
        if random.random() > 0.3:
            train_ids.append(bid)
            X_train.append(reviews_concat)
        else:
            test_ids.append(bid)
            X_test.append(reviews_concat)

    return X_train, X_test, train_ids, test_ids, business_dict, nodes


if __name__ == '__main__':
    read_yelp()
