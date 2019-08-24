import random
import xml.dom.minidom
from collections import defaultdict

from tqdm import tqdm

sample_ratio = 0.02
train_ratio = 0.7
min_per_node = 50


def read_nyt_ids(file_path1, file_path2):
    ids = [[], []]
    random.seed(42)
    for cnt, file_path in enumerate([file_path1, file_path2]):
        filelist_path = './datasets/NYT_annotated_corpus/data/filelist_' + file_path + '.txt'
        with open(filelist_path, 'r') as fin:
            for file_cnt, file_name in tqdm(enumerate(fin)):
                if random.random() < sample_ratio:
                    ids[cnt].append(file_name[:-5])
    return ids


def construct_hierarchy(file_path1, file_path2):
    hierarchy = defaultdict(set)
    for cnt, file_path in enumerate([file_path1, file_path2]):
        print('Processing file %d...' % cnt)
        filelist_path = './datasets/NYT_annotated_corpus/data/filelist_' + file_path + '.txt'
        with open(filelist_path, 'r') as fin:
            for file_cnt, file_name in tqdm(enumerate(fin)):
                file_name = file_name.strip('\n')
                xml_path = './datasets/NYT_annotated_corpus/data/accum' + file_path + '/' + file_name
                try:
                    dom = xml.dom.minidom.parse(xml_path)
                    root = dom.documentElement
                    tags = root.getElementsByTagName('classifier')
                    for tag in tags:
                        type = tag.getAttribute('type')
                        if type != 'taxonomic_classifier':
                            continue
                        hier_path = tag.firstChild.data
                        hier_list = hier_path.split('/')
                        for l in range(1, len(hier_list)):
                            parent = '/'.join(hier_list[:l])
                            child = '/'.join(hier_list[:l + 1])
                            if not child in hierarchy[parent]:
                                hierarchy[parent].add(child)
                            if not child in hierarchy:
                                hierarchy[child] = set()
                except:
                    print('Something went wrong...')
                    continue
    write_hierarchy_to_file(hierarchy, './datasets/nyt/nyt_hier')


def write_hierarchy_to_file(hierarchy, filepath):
    with open(filepath, 'w') as fout:
        nodes = hierarchy.keys()
        for parent in nodes:
            for child in hierarchy[parent]:
                fout.write(parent + '\t' + child + '\n')


def trim_nyt_tree():
    ids_1, ids_2 = read_nyt_ids('1987-02', '2003-07')
    label_cnt = defaultdict(int)
    for id_idx in tqdm(range(len(ids_1) + len(ids_2))):
        if id_idx < len(ids_1):
            doc_id = ids_1[id_idx]
            xml_path = './datasets/NYT_annotated_corpus/data/accum1987-02/' + str(doc_id) + '.xml'
        else:
            doc_id = ids_2[id_idx - len(ids_1)]
            xml_path = './datasets/NYT_annotated_corpus/data/accum2003-07/' + str(doc_id) + '.xml'
        try:
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement
            tags = root.getElementsByTagName('classifier')
            for tag in tags:
                type = tag.getAttribute('type')
                if type != 'taxonomic_classifier':
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split('/')
                for l in range(1, len(hier_list) + 1):
                    label = '/'.join(hier_list[:l])
                    label_cnt[label] += 1
        except:
            print('Something went wrong...')
            continue
    label_cnt = {label: label_cnt[label] for label in label_cnt if label_cnt[label] > min_per_node}
    with open('nyt/nyt_trimed_hier', 'w') as fout:
        for label in label_cnt:
            fout.write(label + '\n')
    return label_cnt


def read_nyt():
    p2c = defaultdict(list)
    id2doc = defaultdict(lambda: defaultdict(list))
    nodes = defaultdict(lambda: defaultdict(list))
    random.seed(42)

    trimmed_nodes = set()
    with open('nyt/nyt_trimed_hier', 'r') as fin:
        for line in fin:
            trimmed_nodes.add(line.strip('\n'))
    with open('nyt/nyt_hier', 'r') as f:
        for line in f:
            line = line.strip('\n')
            parent, child = line.split('\t')
            if parent in trimmed_nodes and child in trimmed_nodes:
                p2c[parent].append(child)
    for label in p2c:
        for children in p2c[label]:
            nodes[label]['children'].append(children)
            nodes[children]['parent'].append(label)

    ids_1, ids_2 = read_nyt_ids('1987-02', '2003-07')
    X_train = []
    X_test = []
    train_ids = []
    test_ids = []
    for id_idx in tqdm(range(len(ids_1) + len(ids_2))):
        if id_idx < len(ids_1):
            doc_id = ids_1[id_idx]
            xml_path = './datasets/NYT_annotated_corpus/data/accum1987-02/' + str(doc_id) + '.xml'
        else:
            doc_id = ids_2[id_idx - len(ids_1)]
            xml_path = './datasets/NYT_annotated_corpus/data/accum2003-07/' + str(doc_id) + '.xml'
        try:
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement
            tags = root.getElementsByTagName('p')
            text = ''
            for tag in tags[1:]:
                text += tag.firstChild.data
            if text == '':
                print('Something went wrong with text...')
                continue
            tags = root.getElementsByTagName('classifier')
            for tag in tags:
                type = tag.getAttribute('type')
                if type != 'taxonomic_classifier':
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split('/')
                for l in range(1, len(hier_list) + 1):
                    label = '/'.join(hier_list[:l])
                    if label in trimmed_nodes and not label in id2doc[doc_id]['categories']:
                        id2doc[doc_id]['categories'].append(label)
            if not doc_id in id2doc:
                continue
            if random.random() < train_ratio:
                train_ids.append(doc_id)
                X_train.append(text)
            else:
                test_ids.append(doc_id)
                X_test.append(text)
        except:
            print('Something went wrong...')
            continue

    return X_train, X_test, train_ids, test_ids, dict(id2doc), dict(nodes)
