import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue

def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    return u2i_index, i2u_index

def dataConversion(dataset_name):
        # 打开文件
    with open('data/%s' % dataset_name, 'r') as file:
    # 逐行读取文件
        user_dictionary = {}
        videoID_user_dictionary = {}
        sideinfo1_dictionary = {}
        sideinfo2_dictionary = {}
        videoID_user_array = []
        sideinfo1_array = []
        sideinfo2_array = []
        for line in file:
        # 使用 split() 方法按空格分割行，并去除换行符
            columns = line.strip().split()
            user_id = get_id(columns[0], user_dictionary)
            for i in columns[1:]:
                j = i.split(':')
                videoID_user = get_id(j[0], videoID_user_dictionary)
                sideinfo1 = get_id(j[1], sideinfo1_dictionary)
                sideinfo2 = get_id(j[2], sideinfo2_dictionary)
                videoID_user_array.append((user_id, videoID_user))
                sideinfo1_array.append((user_id, sideinfo1))
                sideinfo2_array.append((user_id, sideinfo2))
        for i in sideinfo1_array:
            print(i)
        with open('data/videoID_user.txt', 'w') as file:
            for item in videoID_user_array:
                line = ' '.join(map(str, item))
                file.write(f"{str(line)}\n")
        with open('data/sideinfo1.txt', 'w') as file:
            for item in sideinfo1_array:
                line = ' '.join(map(str, item))
                file.write(f"{str(line)}\n")
        with open('data/sideinfo2.txt', 'w') as file:
            for item in sideinfo2_array:
                line = ' '.join(map(str, item))
                file.write(f"{str(line)}\n")
    u2i_index = 1
    i2u_index = 1
    return u2i_index, i2u_index
def get_id(name_id, category):
    if name_id in category:
        return category[name_id]
    else:
        category[name_id] = len(category) + 1
        return category[name_id]



# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, sideinfo1_train, sideinfo2_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        s1 = np.zeros([maxlen], dtype=np.int32)
        s2 = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i in reversed(sideinfo1_train[uid][:-1]):
            s1[idx] = i
            idx -= 1
            if idx == -1: break

        idx = maxlen - 1
        for i in reversed(sideinfo2_train[uid][:-1]):
            s2[idx] = i
            idx -= 1
            if idx == -1: break
        
        return (uid, seq, pos, neg, s1, s2)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, User_sideinfo1, User_sideinfo2, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      User_sideinfo1,
                                                      User_sideinfo2,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    User_sideinfo1 = defaultdict(list)
    User_sideinfo2 = defaultdict(list)
    User = defaultdict(list)
    user_train = {}
    sideinfo1_train = {}
    sideinfo2_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    f1 = open('data/sideinfo1.txt', 'r')
    f2 = open('data/sideinfo2.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)
    for line in f1:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        User_sideinfo1[u].append(i)

    for line in f2:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        User_sideinfo2[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            sideinfo1_train[user] = User_sideinfo1[user]
            sideinfo2_train[user] = User_sideinfo2[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            sideinfo1_train[user] = User_sideinfo2[user][:-2]
            sideinfo2_train[user] = User_sideinfo2[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum, sideinfo1_train, sideinfo2_train]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum, item_genres1, item_genres2] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        genre1 = np.zeros([args.maxlen], dtype=np.int32)
        genre2 = np.zeros([args.maxlen], dtype=np.int32)

        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        idx = args.maxlen - 1
        genre1[idx] = item_genres1[u][0]
        idx -= 1
        for i in reversed(item_genres1[u]):
            genre1[idx] = i
            idx -= 1
            if idx == -1: break

        idx = args.maxlen - 1
        genre2[idx] = item_genres2[u][0]
        idx -= 1
        for i in reversed(item_genres2[u]):
            genre2[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [genre1], [genre2]]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum, item_genres1, item_genres2] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        genre1 = np.zeros([args.maxlen], dtype=np.int32)
        genre2 = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        idx = args.maxlen - 1
        genre1[idx] = item_genres1[u][0]
        idx -= 1
        for i in reversed(item_genres1[u]):
            genre1[idx] = i
            idx -= 1
            if idx == -1: break

        idx = args.maxlen - 1
        genre2[idx] = item_genres2[u][0]
        idx -= 1
        for i in reversed(item_genres2[u]):
            genre2[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx, [genre1], [genre2]]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
