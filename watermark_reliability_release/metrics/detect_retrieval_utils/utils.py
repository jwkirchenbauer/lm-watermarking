import numpy as np
import torch
import random
from collections import Counter

unk_string = "UUUNKKK"

def get_ngrams(examples, share_vocab, max_len=200000, n=3):
    def update_counter(counter, sentence):
        word = " " + sentence.strip() + " "
        lis = []
        for j in range(len(word)):
            idx = j
            ngram = ""
            while idx < j + n and idx < len(word):
                ngram += word[idx]
                idx += 1
            if not len(ngram) == n:
                continue
            lis.append(ngram)
        counter.update(lis)

    counter = Counter()
    if not share_vocab:
        counter_fr = Counter()

    for i in examples:
        update_counter(counter, i[0].sentence)
        if share_vocab:
            update_counter(counter, i[1].sentence)
        else:
            update_counter(counter_fr, i[1].sentence)

    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0:max_len]
    if not share_vocab:
        counter_fr = sorted(counter_fr.items(), key=lambda x: x[1], reverse=True)[0:max_len]

    vocab = {}
    for i in counter:
        vocab[i[0]] = len(vocab)

    vocab[unk_string] = len(vocab)
    if share_vocab:
        return vocab, None

    vocab_fr = {}
    for i in counter_fr:
        vocab_fr[i[0]] = len(vocab_fr)
    vocab_fr[unk_string] = len(vocab_fr)

    return vocab, vocab_fr

def get_words(examples, share_vocab, max_len=200000):
    def update_counter(counter, sentence):
        counter.update(sentence.split())

    counter = Counter()
    if not share_vocab:
        counter_fr = Counter()

    for i in examples:
        update_counter(counter, i[0].sentence)
        if share_vocab:
            update_counter(counter, i[1].sentence)
        else:
            update_counter(counter_fr, i[1].sentence)

    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0:max_len]
    if not share_vocab:
        counter_fr = sorted(counter_fr.items(), key=lambda x: x[1], reverse=True)[0:max_len]

    vocab = {}
    for i in counter:
        vocab[i[0]] = len(vocab)

    vocab[unk_string] = len(vocab)
    if share_vocab:
        return vocab, None

    vocab_fr = {}
    for i in counter_fr:
        vocab_fr[i[0]] = len(vocab_fr)
    vocab_fr[unk_string] = len(vocab_fr)

    return vocab, vocab_fr

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])

    return list(zip(range(len(minibatches)), minibatches))

def max_pool(x, lengths, gpu):
    out = torch.FloatTensor(x.size(0), x.size(2)).zero_()
    if gpu:
        out = out.cuda()
    for i in range(len(lengths)):
        out[i] = torch.max(x[i][0:lengths[i]], 0)[0]
    return out

def mean_pool(x, lengths, gpu):
    out = torch.FloatTensor(x.size(0), x.size(2)).zero_()
    if gpu:
        out = out.cuda()
    for i in range(len(lengths)):
        out[i] = torch.mean(x[i][0:lengths[i]], 0)
    return out

def lookup(words, w, zero_unk):
    if w in words:
        return words[w]
    else:
        if zero_unk:
            return None
        else:
            return words[unk_string]

class Batch(object):
    def __init__(self):
        self.g1 = None
        self.g1_l = None
        self.g2 = None
        self.g2_l = None
        self.p1 = None
        self.p1_l = None
        self.p2 = None
        self.p2_l = None

class BigExample(object):
    def __init__(self, arr, vocab, rev_vocab, scramble_rate):
        self.arr = arr
        self.sentence = [rev_vocab[i] for i in arr]
        self.sentence = " ".join(self.sentence)
        self.embeddings = [i for i in arr]
        if scramble_rate:
            if random.random() <= scramble_rate:
                random.shuffle(self.embeddings)
        if len(self.embeddings) == 0:
            self.embeddings = [vocab[unk_string]]

class Example(object):
    def __init__(self, sentence, lower_case):
        self.sentence = sentence.strip()
        if lower_case:
            self.sentence = self.sentence.lower()
        self.embeddings = []

    def populate_ngrams(self, sentence, words, zero_unk, n):
        sentence = " " + sentence.strip() + " "
        embeddings = []

        for j in range(len(sentence)):
            idx = j
            gr = ""
            while idx < j + n and idx < len(sentence):
                gr += sentence[idx]
                idx += 1
            if not len(gr) == n:
                continue
            wd = lookup(words, gr, zero_unk)
            if wd is not None:
                embeddings.append(wd)

        if len(embeddings) == 0:
            return [words[unk_string]]
        return embeddings

    def populate_embeddings(self, words, zero_unk, ngrams, scramble_rate=0):
        if ngrams:
            self.embeddings = self.populate_ngrams(self.sentence, words, zero_unk, ngrams)
        else:
            arr = self.sentence.split()
            if scramble_rate:
                if random.random() <= scramble_rate:
                    random.shuffle(arr)
            for i in arr:
                wd = lookup(words, i, zero_unk)
                if wd is not None:
                    self.embeddings.append(wd)
            if len(self.embeddings) == 0:
                self.embeddings = [words[unk_string]]
