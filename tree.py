import logging
import pickle
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


class Tree:
    def __init__(self, args, train_ids, test_ids, id2doc=None, id2doc_a=None, nodes=None, X_train=None,
                 X_test=None, rootname=None):
        # child, parent, ancestor
        self.c2p = defaultdict(list)
        self.p2c = defaultdict(list)
        self.p2c_idx = defaultdict(list)
        self.c2p_idx = defaultdict(list)
        self.c2a = defaultdict(set)

        # real id to auto-increment id
        self.id2idx = {}
        self.idx2id = {}
        # real id to label name
        self.id2name = {}
        self.name2id = {}
        # real id to height
        self.id2h = {}
        # doc id to doc obj
        self.id2doc = id2doc
        self.id2doc_ancestors = id2doc_a
        self.nodes = nodes  # for yelp
        self.rootname = rootname
        self.id2doc_h = {}
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.X_train = X_train
        self.X_test = X_test
        self.taken_actions = None
        self.n_update = 0  # global update times
        self.data_cache = None  # loaded from pkl
        self.last_R = None  # R at last time step
        self.id2prob = defaultdict(dict)  # [id][class] = p

        self.args = args
        self.logger = logging.getLogger('exp')
        self.miF = (0, 0)
        self.maF = (0, 0)
        self.cur_epoch = 0
        self.read_mapping()
        self.read_edges()
        for parent in self.p2c:
            for child in self.p2c[parent]:
                self.p2c_idx[self.id2idx[parent]].append(self.id2idx[child])
                self.c2p_idx[self.id2idx[child]].append(self.id2idx[parent])
        self.logger.info('# class = {}'.format(len(self.name2id)))
        self.n_class = len(self.name2id)
        self.class_idx = list(range(1, self.n_class))

        self.p2c_idx_np = self.pad_p2c_idx()
        if args.mode != 'sl' and args.mode != 'hmcn':
            self.next_true_bin, self.next_true = self.generate_next_true(keep_cur=True)
        self.logger.info('{} terms have more than 1 parent'.format(sum([len(v) > 1 for v in self.c2p.values()])))
        leaves = set(self.c2p) - set(self.p2c)
        self.logger.info('{} terms are leaves'.format(len(leaves)))
        for term in self.c2p:
            ancestor_q = [i for i in self.c2p[term]]
            while len(ancestor_q) != 0:
                cur = ancestor_q.pop(0)
                if cur in self.c2p:
                    ancestor_q.extend(self.c2p[cur])
                    # exclude root in one's ancestors
                    self.c2a[term].add(cur)

        # stat info
        # note that only the first path is considered
        for k in self.id2name:
            cur = k
            h = 0
            while cur in self.c2p:
                h += 1
                cur = self.c2p[cur][0]
            self.id2h[k] = h
        if args.stat_check:
            self.logger.info('node height description:')
            self.logger.info(pd.Series(list(self.id2h.values())).describe())
        self.doc_check(self.id2doc, 'all')

    def pad_p2c_idx(self):
        col = max([len(c) for c in self.p2c_idx.values()]) + 2
        res = np.zeros((len(self.name2id), col), dtype=int)
        for row_i in range(len(self.name2id)):
            # stay at cur node
            if self.args.allow_stay:
                res[row_i, len(self.p2c_idx[row_i])] = row_i
                if self.args.allow_up:
                    if row_i in self.c2p_idx:
                        res[row_i, len(self.p2c_idx[row_i]) + 1] = self.c2p_idx[row_i][0]
            # next level node
            res[row_i, :len(self.p2c_idx[row_i])] = self.p2c_idx[row_i]
        return res

    def get_next_candidates(self, last_selections, cur_candidates, nonstop=False):
        all_stop = True
        for candi, sel in zip(cur_candidates, last_selections):
            if sel != 0:  # 0 is not in candi
                candi.remove(sel)
            if sel == self.n_class:  # stop action
                candi.clear()
            else:
                candi.update(self.p2c_idx[sel])
                all_stop = False
            if not nonstop:
                candi.add(self.n_class)
        return cur_candidates, self.pad_candidates(cur_candidates), all_stop

    def update_actions(self, cur_class_batch):
        for taken, a in zip(self.taken_actions, cur_class_batch):
            taken.add(a)

    def pad_candidates(self, cur_candidates):
        col = max([len(c) for c in cur_candidates])
        res = np.zeros((len(cur_candidates), col), dtype=int)
        for row_i, c in enumerate(cur_candidates):
            res[row_i, :len(c)] = list(c)
        return res

    def doc_check(self, docids, name):
        avg_h = []
        n_classes = []
        hs = []
        label_key = 'categories'
        classes = []
        for docid in docids:
            if len(self.id2doc[docid][label_key]) != len(set(self.id2doc[docid][label_key])):
                self.logger.error(label_key)
                self.logger.error(self.id2doc[docid][label_key])
                exit(1)
            classes.extend(self.id2doc[docid][label_key])
            h = [self.id2h[self.name2id[term]] for term in self.id2doc[docid][label_key]]
            hs.append(h)
            self.id2doc_h[docid] = h
            avg_h.append(np.mean(h))
            n_classes.append(len(h))
        if self.args.stat_check:
            self.logger.info('check info of classes for each doc in {}'.format(name))
            self.logger.info('heights of classes')
            self.logger.info(hs[:5])
            self.logger.info('avg_height of labels')
            self.logger.info(pd.Series(avg_h).describe())
            self.logger.info('label count of documents')
            self.logger.info(pd.Series(n_classes).describe())
            self.logger.info('node support')
            self.logger.info(Counter(classes))

    def h_doc_batch(self, docids):
        return [self.id2doc_h[docid] for docid in docids]

    def h_batch(self, ids):
        return [self.id2h[self.idx2id[vid]] for vid in ids]

    def p2c_batch(self, ids):
        # ids is virtual
        res = self.p2c_idx_np[ids]
        # remove columns full of zeros
        return res[:, ~np.all(res == 0, axis=0)]

    def generate_next_true(self, keep_cur=False):
        self.logger.info('keep_cur={}'.format(keep_cur))
        next_true_bin = defaultdict(lambda: defaultdict(list))
        next_true = defaultdict(lambda: defaultdict(list))
        for did in self.id2doc_ancestors:
            class_idx_set = set(self.id2doc_ancestors[did]['class_idx'])
            class_idx_set.add(0)
            for c in class_idx_set:
                for idx, next_c in enumerate(self.p2c_idx[c]):
                    if next_c in class_idx_set:
                        next_true_bin[did][c].append(1)
                        next_true[did][c].append(next_c)
                    else:
                        next_true_bin[did][c].append(0)
                # if lowest label
                if len(next_true[did][c]) == 0:
                    next_true[did][c].append(c)
                    if self.args.allow_stay:
                        next_true_bin[did][c].append(1)
                elif keep_cur and c != 0:
                    # append 1 only for loss calculation
                    next_true_bin[did][c].append(1)
        return next_true_bin, next_true

    def get_next(self, cur_class_batch, next_classes_batch, doc_ids):
        assert len(cur_class_batch) == len(doc_ids)
        next_classes_batch_true = np.zeros(next_classes_batch.shape)
        indices = []
        next_class_batch_true = []
        for ct, (c, did) in enumerate(zip(cur_class_batch, doc_ids)):
            nt = self.next_true_bin[did][c]
            if len(self.next_true[did][c]) == 0:
                print(ct, did, c)
                print(nt, self.next_true[did][c])
                print(self.id2doc_ancestors[did])
                exit(-1)
            next_classes_batch_true[ct][:len(nt)] = nt
            for idx in self.next_true[did][c]:
                indices.append(ct)
                next_class_batch_true.append(idx)
        doc_ids = [doc_ids[idx] for idx in indices]
        return next_classes_batch_true, indices, np.array(next_class_batch_true), doc_ids

    def get_next_by_probs(self, cur_class_batch, next_classes_batch, doc_ids, probs, save_prob):
        assert len(cur_class_batch) == len(doc_ids) == len(doc_ids) == len(probs)
        indices = []
        next_class_batch_pred = []
        if save_prob:
            thres = 0
        else:
            thres = 0.5
        preds = (probs > thres).int().data.cpu().numpy()
        for ct, (c, next_classes, did, pred, p) in enumerate(
                zip(cur_class_batch, next_classes_batch, doc_ids, preds, probs)):
            # need allow_stay=True, filter last one (cur) to avoid duplication
            next_pred = np.nonzero(pred)[0]
            if not self.args.multi_label:
                idx_above_thres = np.argsort(p.data.cpu().numpy()[next_pred])
                for idx in idx_above_thres[::-1]:
                    if next_classes[next_pred[idx]] != c:
                        next_pred = [next_pred[idx]]
                        break
            else:
                if len(next_pred) != 0 and next_classes[next_pred[-1]] == c:
                    next_pred = next_pred[:-1]
            # if no next > threshold, stay at current class
            if len(next_pred) == 0:
                p_selected = []
                next_pred = [c]
            else:
                p_selected = p.data.cpu().numpy()[next_pred]
                next_pred = next_classes[next_pred]
            # indices remember where one is from; idx is virtual class idx
            for idx in next_pred:
                indices.append(ct)
                next_class_batch_pred.append(idx)
            if save_prob:
                for idx, p_ in zip(next_pred, p_selected):
                    if idx in self.id2prob[did]:
                        self.logger.warning(f'[{did}][{idx}] already existed!')
                    self.id2prob[did][idx] = p_
        doc_ids = [doc_ids[idx] for idx in indices]
        return indices, np.array(next_class_batch_pred), doc_ids

    def get_flat_idx_each_layer(self):
        flat_idx_each_layer = [[0]]
        idx2layer_idx = {}
        idx2layer_idx[0] = (0, 0)
        for i in range(self.args.n_steps_sl):
            flat_idx_each_layer.append([])
            current_nodes = flat_idx_each_layer[i]
            for current_node in current_nodes:
                child_nodes = self.p2c_idx[current_node]
                for (in_layer_idx, child_node) in enumerate(child_nodes):
                    flat_idx_each_layer[i + 1].append(child_node)
                    idx2layer_idx[child_node] = (i, in_layer_idx)

        self.flat_idx_each_layer = flat_idx_each_layer
        self.idx2layer_idx = idx2layer_idx
        return flat_idx_each_layer, idx2layer_idx

    def get_layer_node_number(self):
        layer_numer = max(self.id2h.values())
        local_output_number = [0] * layer_numer
        for id in self.id2h:
            if self.id2h[id] == 0:
                continue
            local_output_number[self.id2h[id] - 1] += 1
        assert sum(local_output_number) == self.n_class - 1
        return local_output_number

    def calc_reward(self, notLast, actions, ids):
        if self.args.reward == '01':
            return self.calc_reward_01(actions, ids)
        elif self.args.reward == 'f1':
            return self.calc_reward_f1(actions, ids)
        elif self.args.reward == '01-1':
            return self.calc_reward_neg(actions, ids)
        elif self.args.reward == 'direct':
            return self.calc_reward_direct(notLast, actions, ids)
        elif self.args.reward == 'taxo':
            return self.calc_reward_taxo(actions, ids)
        else:
            raise NotImplementedError

    def calc_reward_f1(self, actions, ids):
        R = []
        for a, i, taken in zip(actions, ids, self.taken_actions):
            taken.add(a)
            y_l_a = self.id2doc_ancestors[i]['class_idx']
            correct = set(y_l_a) & taken
            p = len(correct) / len(taken)
            r = len(correct) / len(y_l_a)
            f1 = 2 * p * r / (p + r + 1e-32)
            R.append(f1)
        R = np.array(R)
        res = np.copy(R)
        if self.last_R is not None:
            res -= self.last_R
        self.last_R = R
        return res

    def calc_reward_direct(self, notLast, actions, ids):
        if notLast:
            return [0] * len(actions)
        R = []
        for a, i, taken in zip(actions, ids, self.taken_actions):
            if a in self.id2doc[i]['class_idx']:
                R.append(1)
            else:
                R.append(0)
        return R

    def calc_reward_taxo(self, actions, ids):
        R = []
        for taken, a, i in zip(self.taken_actions, actions, ids):
            if a in self.id2doc_ancestors[i]['class_idx']:
                R.append(1)
            elif a == self.n_class:
                R.append(0)
            else:
                R.append(-1)
        return R

    def calc_reward_01(self, actions, ids):
        R = []
        label_key = 'categories'
        for a, i, taken in zip(actions, ids, self.taken_actions):
            if a in self.id2doc[i]['class_idx']:
                R.append(1)
                continue
            if a in taken:
                R.append(0)
                continue
            for label in self.id2doc[i][label_key]:
                if self.idx2id[a] in self.c2a[self.name2id[label]]:
                    R.append(1)
                    break
            else:
                R.append(0)
            taken.add(a)
        return R

    def calc_reward_neg(self, actions, ids):
        # actions : virtual id
        # ids: doc id for mesh, yelp | real labels for review
        R = []
        label_key = 'categories'
        assert len(actions) == len(ids) == len(self.taken_actions)
        if self.args.dataset in ['mesh', 'yelp', 'rcv1']:
            for a, i, taken in zip(actions, ids, self.taken_actions):
                if a in self.id2doc[i]['class_idx']:
                    R.append(1)
                    continue
                for label in self.id2doc[i][label_key]:
                    if self.idx2id[a] in self.c2a[self.name2id[label]]:
                        if a in taken:
                            R.append(0)
                        else:
                            R.append(1)
                        break
                else:
                    R.append(-1)
                taken.add(a)
        return R

    def calc_f1(self, pred_l, id_l, save_path=None, output=True):
        assert len(pred_l) == len(id_l)
        if not output:
            mlb = MultiLabelBinarizer(classes=self.class_idx)
            y_l_a = [self.id2doc_ancestors[docid]['class_idx'] for docid in id_l]
            y_true_a = mlb.fit_transform(y_l_a)
            y_pred = mlb.transform(pred_l)
            f1_a = f1_score(y_true_a, y_pred, average='micro')
            f1_a_macro = f1_score(y_true_a, y_pred, average='macro')
            f1_a_s = f1_score(y_true_a, y_pred, average='samples')
            self.logger.info(f'micro:{f1_a:.4f} macro:{f1_a_macro:.4f} samples:{f1_a_s:.4f}')
            return f1_a, f1_a_macro, f1_a_s
        # pred_l_a contains all ancestors (except root)
        pred_l_a = []
        for pred in pred_l:
            pred_l_a.append([])
            for i in pred:
                pred_l_a[-1].append(i)
                # TODO can be refactored
                pred_l_a[-1].extend(
                    [self.id2idx[self.name2id[term]] for term in self.c2a[self.id2name[self.idx2id[i]]]])
        y_l = [self.id2doc[docid]['class_idx'] for docid in id_l]
        if self.id2doc_ancestors:
            y_l_a = [self.id2doc_ancestors[docid]['class_idx'] for docid in id_l]
        else:
            y_l_a = y_l

        self.logger.info('measuring f1 of {} samples...'.format(len(pred_l)))
        avg_len = np.mean([len(p) for p in pred_l])
        self.logger.info(f'len(pred): {avg_len}')
        # self.logger.info(
        #     'docid:{} pred:{} real:{} pred_a:{} real_a:{}'.format(id_l[:5], pred_l[:5], y_l[:5], pred_l_a[:5],
        #                                                           y_l_a[:5]))
        # self.logger.info('Counter(pred_l):{}'.format(Counter(flatten(pred_l))))
        # self.logger.info('Counter(y_l):{}'.format(Counter(flatten(y_l))))
        # self.logger.info('Counter(y_l_a):{}'.format(Counter(flatten(y_l_a))))
        # self.logger.info(self.idx2id)

        calc_leaf_only = False
        if calc_leaf_only:
            y_true_leaf = [[idx for idx in i if len(self.p2c_idx[idx]) == 0] for i in y_l]
            y_pred_leaf = [[idx for idx in i if len(self.p2c_idx[idx]) == 0] for i in pred_l]
            mlb = MultiLabelBinarizer()
            y_true = sparse.csr_matrix(mlb.fit_transform(y_true_leaf))
            print(len(mlb.classes_), 36504)
            y_pred = sparse.csr_matrix(mlb.transform(y_pred_leaf))
            print(f1_score(y_true, y_pred, average='micro'))
            print(f1_score(y_true, y_pred, average='macro'))
            print(f1_score(y_true, y_pred, average='samples'))

        mlb = MultiLabelBinarizer(classes=self.class_idx)
        y_true = sparse.csr_matrix(mlb.fit_transform(y_l))
        y_pred = sparse.csr_matrix(mlb.transform(pred_l))
        f1 = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')

        calc_other_f1 = True
        if calc_other_f1:
            y_true_a = sparse.csr_matrix(mlb.transform(y_l_a))
            y_pred_a = sparse.csr_matrix(mlb.transform(pred_l_a))
            if save_path is not None:
                self.logger.info('saving to {}/preds.pkl'.format(save_path))
                pickle.dump((y_pred, id_l, y_true_a), open('{}/preds.pkl'.format(save_path), 'wb'))
            f1_a = f1_score(y_true_a, y_pred, average='micro')
            f1_aa = f1_score(y_true_a, y_pred_a, average='micro')
            f1_a_macro = f1_score(y_true_a, y_pred, average='macro')
            f1_aa_macro = f1_score(y_true_a, y_pred_a, average='macro')
            f1_aa_s = f1_score(y_true_a, y_pred_a, average='samples')
            # from sklearn.metrics import classification_report
            # print(classification_report(y_true, y_pred))
        else:
            f1_a = 0
            f1_aa = 0
            f1_a_macro = 0
            f1_aa_macro = 0
            f1_aa_s = 0
        return f1, f1_a, f1_aa, f1_macro, f1_a_macro, f1_aa_macro, f1_aa_s

    def acc(self, pred_l, id_l):
        R = []
        for pred, i in zip(pred_l, id_l):
            R.append(int(pred in self.id2doc[i]['class_idx']))
        return np.mean(R)

    def acc_multi(self, pred_l, id_l):
        R = []
        for preds, i in zip(pred_l, id_l):
            for pred in preds:
                R.append(int(pred in self.id2doc[i]['class_idx']))
        return np.mean(R)

    def read_mapping(self):
        for idx, c in enumerate(self.nodes):
            if c == self.rootname:
                root_idx = idx
            self.id2name[c] = c
            self.name2id[c] = c
            self.id2idx[c] = idx
            self.idx2id[idx] = c
        # put root to idx 0
        self.idx2id[root_idx] = self.idx2id[0]
        self.id2idx[self.idx2id[root_idx]] = root_idx
        self.id2idx[self.rootname] = 0
        self.idx2id[0] = self.rootname
        # remove root from labels and add class_idx
        for bid in self.id2doc:
            self.id2doc[bid]['categories'] = [c for c in self.id2doc[bid]['categories'] if c != self.rootname]
            self.id2doc[bid]['class_idx'] = [self.id2idx[c] for c in self.id2doc[bid]['categories']]
        if self.id2doc_ancestors is None:
            return
        for bid in self.id2doc_ancestors:
            self.id2doc_ancestors[bid]['categories'] = [c for c in self.id2doc_ancestors[bid]['categories'] if
                                                        c != self.rootname]
            self.id2doc_ancestors[bid]['class_idx'] = [self.id2idx[c] for c in self.id2doc_ancestors[bid]['categories']]

    def read_edges(self):
        for parent in self.nodes:
            for child in self.nodes[parent]['children']:
                self.c2p[child].append(parent)
                self.p2c[parent].append(child)

    def remove_stop(self):
        for taken in self.taken_actions:
            taken.discard(self.n_class)
