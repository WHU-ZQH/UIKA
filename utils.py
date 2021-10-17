# -*- coding: utf-8 -*-
import numpy as np
import pickle
import os
from pytorch_pretrained_bert import BertModel, BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained(
    r'D:\BERT\bert-base-uncased\vocab.txt')  # bert model path


def pad_dataset(dataset, bs):
    n_records = len(dataset)
    n_padded = bs - n_records % bs
    new_dataset = [t for t in dataset]
    new_dataset.extend(dataset[:n_padded])
    return new_dataset


def pad_seq(dataset, field, max_len, symbol):
    n_records = len(dataset)
    for i in range(n_records):
        assert isinstance(dataset[i][field], list)
        while len(dataset[i][field]) < max_len:
            dataset[i][field].append(symbol)
    return dataset


def read(path):
    dataset = []
    sid = 0  # id
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            find_label = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 0
                    elif '/n' in t:
                        end = '/n'
                        y = 1
                    elif '/0' in t:
                        end = '/0'
                        y = 2
                    words.append(t.strip(end))
                    target_words.append(t.strip(end))

                    if not find_label:
                        find_label = True
                        record['y'] = y
                        left_most = right_most = tokens.index(t)
                    else:
                        right_most += 1
                else:
                    words.append(t)
            if not find_label:
                record['y'] = None
            for pos in range(len(tokens)):
                if pos < left_most:
                    d.append(right_most - pos)
                else:
                    d.append(pos - left_most)
            record['sent'] = line.strip()
            record['words'] = words.copy()
            record['twords'] = target_words.copy()
            record['wc'] = len(words)
            record['wct'] = len(record['twords'])
            record['dist'] = d.copy()
            record['sid'] = sid
            record['beg'] = left_most
            record['end'] = right_most + 1
            sid += 1
            if record['y'] is not None:
                dataset.append(record)
    return dataset


def load_data(ds_name):
    data_npz = 'dataset_npy/dataset_%s.npz' % ds_name
    vocab_npy = 'dataset_npy/vocab_%s.npy' % ds_name
    if not os.path.exists(data_npz):
        train_file = './dataset/%s/train.txt' % ds_name
        test_file = './dataset/%s/test.txt' % ds_name
        train_set = read(path=train_file)
        test_set = read(path=test_file)
        train_wc = [t['wc'] for t in train_set]
        test_wc = [t['wc'] for t in test_set]
        max_len = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc)
        train_t_wc = [t['wct'] for t in train_set]
        test_t_wc = [t['wct'] for t in test_set]
        max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)
        train_set = pad_seq(dataset=train_set, field='dist', max_len=max_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='dist', max_len=max_len, symbol=-1)
        train_set = calculate_position_weight(dataset=train_set)
        test_set = calculate_position_weight(dataset=test_set)
        vocab = build_vocab(dataset=train_set + test_set)
        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_len)
        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)
        dataset = [train_set, test_set]
        np.savez(data_npz, train=train_set, test=test_set)
        np.save(vocab_npy, vocab)
    else:
        dataset = np.load(data_npz, allow_pickle=True)
        train_set, test_set = dataset['train'], dataset['test']
        train_set, test_set = train_set.tolist(), test_set.tolist()
        dataset = [train_set, test_set]
        vocab = np.load(vocab_npy, allow_pickle=True).tolist()
    return dataset, vocab


def read_bert(path):
    dataset = []
    sid = 0  # id
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            tokens = line.strip().split()
            words, target_words = [], []
            d = []
            find_label = False
            for t in tokens:
                if '/p' in t or '/n' in t or '/0' in t:
                    end = 'xx'
                    y = 0
                    if '/p' in t:
                        end = '/p'
                        y = 0
                    elif '/n' in t:
                        end = '/n'
                        y = 1
                    elif '/0' in t:
                        end = '/0'
                        y = 2
                    words.append(t.strip(end))
                    target_words.append(t.strip(end))

                    if not find_label:
                        find_label = True
                        record['y'] = y
                        left_most = right_most = tokens.index(t)
                    else:
                        right_most += 1
                else:
                    words.append(t)
            if not find_label:
                record['y'] = None
            for pos in range(len(tokens)):
                if pos < left_most:
                    d.append(right_most - pos)
                else:
                    d.append(pos - left_most)

            bert_sentence = bert_tokenizer.tokenize(' '.join(words.copy()))
            bert_aspect = bert_tokenizer.tokenize(' '.join(target_words.copy()))
            record['bert_token'] = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_sentence + ['[SEP]'])
            record['bert_token_aspect'] = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_aspect + ['[SEP]'])
            record['sent'] = line.strip()
            record['words'] = words.copy()
            record['twords'] = target_words.copy()
            record['wc'] = len(words)
            record['wct'] = len(record['twords'])
            record['bert_len'] = len(record['bert_token'])
            record['bert_as_len'] = len(record['bert_token_aspect'])
            record['dist'] = d.copy()
            record['dist'].append(-1)
            record['dist'].insert(0, -1)
            record['sid'] = sid
            record['beg'] = left_most
            record['end'] = right_most + 1
            sid += 1
            if record['y'] is not None:
                dataset.append(record)
    return dataset


def load_data_bert(ds_name):
    data_npz = 'dataset_npy/dataset_%s_bert.npz' % ds_name
    vocab_npy = 'dataset_npy/vocab_%s_bert.npy' % ds_name
    if not os.path.exists(data_npz):
        train_file = './dataset/%s/train.txt' % ds_name
        test_file = './dataset/%s/test.txt' % ds_name
        train_set = read_bert(path=train_file)
        test_set = read_bert(path=test_file)
        train_t_wc = [t['wct'] for t in train_set]
        test_t_wc = [t['wct'] for t in test_set]
        max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)
        train_bert_len = [t['bert_len'] for t in train_set]
        test_bert_len = [t['bert_len'] for t in test_set]
        max_bert_len = max(train_bert_len) if max(train_bert_len) > max(test_bert_len) else max(test_bert_len)
        train_bert_as_len = [t['bert_as_len'] for t in train_set]
        test_bert_as_len = [t['bert_as_len'] for t in test_set]
        max_bert_as_len = max(train_bert_as_len) if max(train_bert_as_len) > max(test_bert_as_len) else max(
            test_bert_as_len)
        print(max_bert_len, max_bert_as_len)
        num1 = len(train_set)
        num2 = len(test_set)
        for i in range(num1):
            train_set[i]['bert_token'].extend([0] * (max_bert_len - len(train_set[i]['bert_token'])))
            train_set[i]['bert_token_aspect'].extend([0] * (max_bert_as_len - len(train_set[i]['bert_token_aspect'])))
        for i in range(num2):
            test_set[i]['bert_token'].extend([0] * (max_bert_len - len(test_set[i]['bert_token'])))
            test_set[i]['bert_token_aspect'].extend([0] * (max_bert_as_len - len(test_set[i]['bert_token_aspect'])))
        train_set = pad_seq(dataset=train_set, field='dist', max_len=max_bert_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='dist', max_len=max_bert_len, symbol=-1)
        train_set = calculate_position_weight(dataset=train_set)
        test_set = calculate_position_weight(dataset=test_set)
        vocab = build_vocab(dataset=train_set + test_set)
        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_bert_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_bert_len)
        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)
        dataset = [train_set, test_set]
        np.savez(data_npz, train=train_set, test=test_set)
        np.save(vocab_npy, vocab)
    else:
        dataset = np.load(data_npz, allow_pickle=True)
        train_set, test_set = dataset['train'], dataset['test']
        train_set, test_set = train_set.tolist(), test_set.tolist()
        dataset = [train_set, test_set]
        vocab = np.load(vocab_npy, allow_pickle=True).tolist()
    return dataset, vocab


def read_pre(path):
    dataset = []
    sid = 0
    is_save = True
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            words = []
            t, _, text = line.strip().partition(',')
            y, _, text = text.strip().partition(',')
            tokens = text.strip().split()
            for i in tokens:
                words.append(i)
            try:
                if int(y) == 1 or int(y) == 0:
                    record['y'] = int(y)
                else:
                    is_save = False
            except ValueError:
                continue
            if is_save:
                record['sent'] = text
                record['twords'] = list(t.strip().split())
                record['words'] = words.copy()
                record['wc'] = len(words)
                record['sid'] = sid
                sid += 1
                dataset.append(record)
            is_save = True
    return dataset


def load_data_pre(ds_name='Amazon'):
    data_npz = 'dataset_npy/dataset_%s_pre.npz' % ds_name
    vocab_npy = 'dataset_npy/vocab_%s_pre.npy' % ds_name
    if not os.path.exists(data_npz):
        train_file = './dataset/%s/train.txt' % ds_name
        test_file = './dataset/%s/test.txt' % ds_name
        train_set = read_pre(path=train_file)
        test_set = read_pre(path=test_file)
        train_wc = [t['wc'] for t in train_set]
        test_wc = [t['wc'] for t in test_set]
        max_len = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc)
        vocab = build_vocab(dataset=train_set + test_set)
        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_len)
        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=1)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=1)
        dataset = [train_set, test_set]
        np.savez(data_npz, train=train_set, test=test_set)
        np.save(vocab_npy, vocab)
    else:
        dataset = np.load(data_npz, allow_pickle=True)
        train_set, test_set = dataset['train'], dataset['test']
        train_set, test_set = train_set.tolist(), test_set.tolist()
        dataset = [train_set, test_set]
        vocab = np.load(vocab_npy, allow_pickle=True).tolist()
    return dataset, vocab


def read_pre_bert(path):
    dataset = []
    sid = 0
    is_save = True
    with open(path, encoding='utf-8') as fp:
        for line in fp:
            record = {}
            words = []
            t, _, text = line.strip().partition(',')
            y, _, text = text.strip().partition(',')
            tokens = text.strip().split()
            for i in tokens:
                words.append(i)
            try:
                if int(y) == 1 or int(y) == 0:
                    record['y'] = int(y)
                else:
                    is_save = False
            except ValueError:
                continue
            if is_save:
                bert_sentence = bert_tokenizer.tokenize(' '.join(words.copy()))
                bert_aspect = bert_tokenizer.tokenize(t.strip())
                record['bert_token'] = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_sentence + ['[SEP]'])
                record['bert_token_aspect'] = bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + bert_aspect + ['[SEP]'])
                record['bert_len'] = len(record['bert_token'])
                record['bert_as_len'] = len(record['bert_token_aspect'])
                record['sent'] = text
                record['twords'] = list(t.strip().split())
                record['words'] = words.copy()
                record['wc'] = len(words)  # word count
                record['sid'] = sid
                sid += 1
                dataset.append(record)
            is_save = True
    return dataset


def load_data_pre_bert(ds_name='Amazon'):
    data_npz = 'dataset_npy/dataset_%s_pre_bert.npz' % ds_name
    vocab_npy = 'dataset_npy/vocab_%s_pre_bert.npy' % ds_name
    if not os.path.exists(data_npz):
        train_file = './dataset/%s/train.txt' % ds_name
        test_file = './dataset/%s/test.txt' % ds_name
        train_set = read_pre_bert(path=train_file)
        test_set = read_pre_bert(path=test_file)
        train_bert_len = [t['bert_len'] for t in train_set]
        test_bert_len = [t['bert_len'] for t in test_set]
        train_bert_as_len = [t['bert_as_len'] for t in train_set]
        test_bert_as_len = [t['bert_as_len'] for t in test_set]
        bert_len = np.array(train_bert_len + test_bert_len)
        max_bert_len = int(np.mean(bert_len) + 1 * np.std(bert_len))
        bert_as_len = np.array(train_bert_as_len + test_bert_as_len)
        max_bert_as_len = int(np.mean(bert_as_len) + 3 * np.std(bert_as_len))
        print(max_bert_len, max_bert_as_len)
        num1 = len(train_set)
        num2 = len(test_set)
        for i in range(num1):
            if len(train_set[i]['bert_token']) < max_bert_len:
                train_set[i]['bert_token'].extend([0] * (max_bert_len - len(train_set[i]['bert_token'])))
            else:
                train_set[i]['bert_token'] = train_set[i]['bert_token'][:max_bert_len]
            if len(train_set[i]['bert_token_aspect']) < max_bert_as_len:
                train_set[i]['bert_token_aspect'].extend(
                    [0] * (max_bert_as_len - len(train_set[i]['bert_token_aspect'])))
            else:
                train_set[i]['bert_token_aspect'] = train_set[i]['bert_token_aspect'][:max_bert_as_len]
        for i in range(num2):
            if len(test_set[i]['bert_token']) < max_bert_len:
                test_set[i]['bert_token'].extend([0] * (max_bert_len - len(test_set[i]['bert_token'])))
            else:
                test_set[i]['bert_token'] = test_set[i]['bert_token'][:max_bert_len]
            if len(test_set[i]['bert_token_aspect']) < max_bert_as_len:
                test_set[i]['bert_token_aspect'].extend([0] * (max_bert_as_len - len(test_set[i]['bert_token_aspect'])))
            else:
                test_set[i]['bert_token_aspect'] = test_set[i]['bert_token_aspect'][:max_bert_as_len]
        vocab = build_vocab(dataset=train_set + test_set)
        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_bert_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_bert_len)
        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=1)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=1)
        dataset = [train_set, test_set]
        np.savez(data_npz, train=train_set, test=test_set)
        np.save(vocab_npy, vocab)
    else:
        dataset = np.load(data_npz, allow_pickle=True)
        train_set, test_set = dataset['train'], dataset['test']
        train_set, test_set = train_set.tolist(), test_set.tolist()
        dataset = [train_set, test_set]
        vocab = np.load(vocab_npy, allow_pickle=True).tolist()
    return dataset, vocab


def build_vocab(dataset):
    vocab = {}
    idx = 1
    n_records = len(dataset)
    for i in range(n_records):
        for w in dataset[i]['words']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
        for w in dataset[i]['twords']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab


def build_vocab_pre(dataset):
    vocab = {}
    idx = 1
    n_records = len(dataset)
    for i in range(n_records):
        for w in dataset[i]['words']:
            if w not in vocab:
                vocab[w] = idx
                idx += 1
    return vocab


def set_wid(dataset, vocab, max_len):
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['words']
        dataset[i]['wids'] = word2id(vocab, sent, max_len)
    return dataset


def set_tid(dataset, vocab, max_len):
    n_records = len(dataset)
    for i in range(n_records):
        sent = dataset[i]['twords']
        dataset[i]['tids'] = word2id(vocab, sent, max_len)
    return dataset


def word2id(vocab, sent, max_len):
    wids = []
    for w in sent:
        try:
            wids.append(vocab[w])
        except KeyError:
            wids.append(0)
    # wids = [vocab[w] for w in sent]
    if len(wids) > max_len:
        wids = wids[:max_len]
    while len(wids) < max_len:
        wids.append(0)
    return wids


def get_embedding(vocab, ds_name):
    emb_file = "../glove.840B.300d.txt"
    pkl = 'embeddings/%s_840B.pkl' % ds_name
    n_emb = 0
    if not os.path.exists(pkl):
        embeddings = np.zeros((len(vocab) + 1, 300), dtype='float32')
        with open(emb_file, encoding='utf-8') as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                n_emb += 1
                if w in vocab:
                    try:
                        embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                    except ValueError:
                        pass
        pickle.dump(embeddings, open(pkl, 'wb'))
    else:
        embeddings = pickle.load(open(pkl, 'rb'))
    return embeddings


def get_embedding_pre(vocab, ds_name):
    emb_file = "../glove.840B.300d.txt"
    pkl = './embeddings/%s_840B_pre.pkl' % ds_name
    n_emb = 0
    if not os.path.exists(pkl):
        embeddings = np.zeros((len(vocab) + 1, 300), dtype='float32')
        with open(emb_file, encoding='utf-8') as fp:
            for line in fp:
                eles = line.strip().split()
                w = eles[0]
                n_emb += 1
                if w in vocab:
                    try:
                        embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                    except ValueError:
                        pass
        pickle.dump(embeddings, open(pkl, 'wb'))
    else:
        embeddings = pickle.load(open(pkl, 'rb'))
    return embeddings


def build_dataset(ds_name, bs, train_pre=False, is_bert=False):
    if train_pre:
        if is_bert:
            dataset, vocab = load_data_pre_bert(ds_name=ds_name)  ##Amazon or Yelp
        else:
            dataset, vocab = load_data_pre(ds_name=ds_name)
    else:
        if is_bert:
            dataset, vocab = load_data_bert(ds_name=ds_name)
        else:
            dataset, vocab = load_data(ds_name=ds_name)
    if is_bert is False:
        if train_pre:
            embeddings = get_embedding_pre(vocab, ds_name)
        else:
            embeddings = get_embedding(vocab, ds_name)
        for i in range(len(embeddings)):
            if i and np.count_nonzero(embeddings[i]) == 0:
                embeddings[i] = np.random.uniform(-0.25, 0.25, embeddings.shape[1])
        embeddings = np.array(embeddings, dtype='float32')
    train_set = pad_dataset(dataset=dataset[0], bs=bs)
    test_set = pad_dataset(dataset=dataset[1], bs=bs)
    if is_bert:
        return [train_set, test_set]
    else:
        return [train_set, test_set], embeddings


def calculate_position_weight(dataset):
    tmax = 40
    n_tuples = len(dataset)
    for i in range(n_tuples):
        dataset[i]['pw'] = []
        weights = []
        for w in dataset[i]['dist']:
            if w == -1:
                weights.append(0.0)
            elif w > tmax:
                weights.append(0.0)
            else:
                weights.append(1.0 - float(w) / tmax)
        dataset[i]['pw'].extend(weights)
    return dataset
