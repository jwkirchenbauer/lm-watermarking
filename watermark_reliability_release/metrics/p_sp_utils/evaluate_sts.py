import sentencepiece as spm
import os
import io
import numpy as np
import logging
from sacremoses import MosesTokenizer
import random

from scipy.stats import spearmanr, pearsonr


unk_string = "UUUNKKK"

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def lookup(words, w, zero_unk):
    if w in words:
        return words[w]
    else:
        if zero_unk:
            return None
        else:
            return words[unk_string]

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

class STSEval(object):
    def loadFile(self, fpath):
        self.data = {}
        self.samples = []

        for dataset in self.datasets:
            sent1, sent2 = zip(*[l.split("\t") for l in
                               io.open(fpath + '/STS.input.%s.txt' % dataset,
                                       encoding='utf8').read().splitlines()])
            raw_scores = np.array([x for x in
                                   io.open(fpath + '/STS.gs.%s.txt' % dataset,
                                           encoding='utf8')
                                   .read().splitlines()])
            not_empty_idx = raw_scores != ''

            gs_scores = [float(x) for x in raw_scores[not_empty_idx]]
            sent1 = np.array([s.split() for s in sent1])[not_empty_idx]
            sent2 = np.array([s.split() for s in sent2])[not_empty_idx]
            # sort data by length to minimize padding in batcher
            sorted_data = sorted(zip(sent1, sent2, gs_scores),
                                 key=lambda z: (len(z[0]), len(z[1]), z[2]))
            sent1, sent2, gs_scores = map(list, zip(*sorted_data))

            self.data[dataset] = (sent1, sent2, gs_scores)
            self.samples += sent1 + sent2

    def do_prepare(self):
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def run(self, params, batcher):
        results = {}
        for dataset in self.datasets:
            sys_scores = []
            input1, input2, gs_scores = self.data[dataset]
            for ii in range(0, len(gs_scores), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)

            results[self.name + "." + dataset] = {'pearson': pearsonr(sys_scores, gs_scores),
                                'spearman': spearmanr(sys_scores, gs_scores),
                                'nsamples': len(sys_scores)}
            logging.debug('%s : pearson = %.4f, spearman = %.4f' %
                          (dataset, results[self.name + "." + dataset]['pearson'][0],
                           results[self.name + "." + dataset]['spearman'][0]))

        weights = [results[dset]['nsamples'] for dset in results.keys()]
        list_prs = np.array([results[dset]['pearson'][0] for
                            dset in results.keys()])
        list_spr = np.array([results[dset]['spearman'][0] for
                            dset in results.keys()])

        avg_pearson = np.average(list_prs)
        avg_spearman = np.average(list_spr)
        wavg_pearson = np.average(list_prs, weights=weights)
        wavg_spearman = np.average(list_spr, weights=weights)

        results[self.name + "." + 'all'] = {'pearson': {'mean': avg_pearson,
                                      'wmean': wavg_pearson},
                          'spearman': {'mean': avg_spearman,
                                       'wmean': wavg_spearman}}
        logging.debug('ALL (weighted average) : Pearson = %.4f, \
            Spearman = %.4f' % (wavg_pearson, wavg_spearman))
        logging.debug('ALL (average) : Pearson = %.4f, \
            Spearman = %.4f\n' % (avg_pearson, avg_spearman))

        return results


class STS12Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS12 *****\n\n')
        self.seed = seed
        self.datasets = ['MSRpar', 'MSRvid', 'SMTeuroparl',
                         'surprise.OnWN', 'surprise.SMTnews']
        self.loadFile(taskpath)
        self.name = "STS12"


class STS13Eval(STSEval):
    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS13 (-SMT) *****\n\n')
        self.seed = seed
        self.datasets = ['FNWN', 'headlines', 'OnWN', 'SMT']
        self.loadFile(taskpath)
        self.name = "STS13"

class STS14Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS14 *****\n\n')
        self.seed = seed
        self.datasets = ['deft-forum', 'deft-news', 'headlines',
                         'images', 'OnWN', 'tweet-news']
        self.loadFile(taskpath)
        self.name = "STS14"

class STS15Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS15 *****\n\n')
        self.seed = seed
        self.datasets = ['answers-forums', 'answers-students',
                         'belief', 'headlines', 'images']
        self.loadFile(taskpath)
        self.name = "STS15"

class STS16Eval(STSEval):
    def __init__(self, taskpath, seed=1111):
        logging.debug('***** Transfer task : STS16 *****\n\n')
        self.seed = seed
        self.datasets = ['answer-answer', 'headlines', 'plagiarism',
                         'postediting', 'question-question']
        self.loadFile(taskpath)
        self.name = "STS16"

class STSBenchmarkEval(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSBenchmark*****\n\n')
        self.seed = seed
        dev = self.loadFile(os.path.join(task_path, 'sts-dev.csv'))
        test = self.loadFile(os.path.join(task_path, 'sts-test.csv'))
        self.data = {}
        self.data['dev'] = dev
        self.data['test'] = test
        self.datasets = ["dev", "test"]
        self.name = "Benchmark"

    def loadFile(self, fpath):
        gs_scores = []
        sent1 = []
        sent2 = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sent1.append(text[5].split())
                sent2.append(text[6].split())
                gs_scores.append(float(text[4]))

        sorted_data = sorted(zip(sent1, sent2, gs_scores),
                    key=lambda z: (len(z[0]), len(z[1]), z[2]))

        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        return sent1, sent2, gs_scores

class STSHard(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : STSHard*****\n\n')
        self.seed = seed
        hard_pos = self.loadFile(os.path.join(task_path, 'hard-pos.txt'))
        hard_neg = self.loadFile(os.path.join(task_path, 'hard-neg.txt'))
        self.data = {}
        self.data['hard-pos'] = hard_pos
        self.data['hard-neg'] = hard_neg
        self.datasets = ["hard-pos", "hard-neg"]
        self.name = "Hard"

    def loadFile(self, fpath):
        gs_scores = []
        sent1 = []
        sent2 = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                sent1.append(text[0].split())
                sent2.append(text[1].split())
                gs_scores.append(float(text[2]))

        sorted_data = sorted(zip(sent1, sent2, gs_scores),
                    key=lambda z: (len(z[0]), len(z[1]), z[2]))

        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        return sent1, sent2, gs_scores

class SemEval17(STSEval):
    def __init__(self, task_path, seed=1111):
        logging.debug('\n\n***** Transfer task : SemEval17*****\n\n')
        self.seed = seed
        self.data = {}
        self.datasets = ["STS.input.track1.ar-ar.txt",
                         "STS.input.track2.ar-en.txt",
                         "STS.input.track3.es-es.txt",
                         "STS.input.track4a.es-en.txt",
                         "STS.input.track5.en-en.txt",
                         "STS.input.track6.tr-en.txt"]

        for i in self.datasets:
            self.data[i] = self.loadFile(os.path.join(task_path, i))

        self.name = "SemEval17"

    def loadFile(self, fpath):
        gs_scores = []
        sent1 = []
        sent2 = []
        with io.open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip().split('\t')
                if len(text) != 3:
                    continue
                sent1.append(text[0].split())
                sent2.append(text[1].split())
                gs_scores.append(float(text[2]))

        sorted_data = sorted(zip(sent1, sent2, gs_scores),
                    key=lambda z: (len(z[0]), len(z[1]), z[2]))

        sent1, sent2, gs_scores = map(list, zip(*sorted_data))

        return sent1, sent2, gs_scores

def batcher(params, batch):
    batch = [" ".join(s) for s in batch]
    new_batch = []
    for p in batch:
        if params.tokenize:
            tok = params.entok.tokenize(p, escape=False)
            p = " ".join(tok)
        if params.lower_case:
            p = p.lower()
        if params.model.args.debug:
            print("Logging STS: {0}".format(p))
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        p = Example(p, params.lower_case)
        p.populate_embeddings(params.model.vocab, params.model.zero_unk, params.model.ngrams)
        new_batch.append(p)
    x, l = params.model.torchify_batch(new_batch)
    vecs = params.model.encode(x, l)
    return vecs.detach().cpu().numpy()

def evaluate_sts(model, params):

    sp = spm.SentencePieceProcessor()
    sp.Load(params.sp_model)

    entok = MosesTokenizer(lang='en')

    from argparse import Namespace

    args = Namespace(batch_size=32, entok=entok, sp=sp,
                     params=params, model=model, lower_case=params.lower_case,
                     tokenize=params.tokenize)

    s = STS12Eval('STS/STS12-en-test')
    s.do_prepare()
    results = s.run(args, batcher)
    s = STS13Eval('STS/STS13-en-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STS14Eval('STS/STS14-en-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STS15Eval('STS/STS15-en-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STS16Eval('STS/STS16-en-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = SemEval17('STS/STS17-test')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STSBenchmarkEval('STS/STSBenchmark')
    s.do_prepare()
    results.update(s.run(args, batcher))
    s = STSHard('STS/STSHard')
    s.do_prepare()
    results.update(s.run(args, batcher))

    for i in results:
        print(i, results[i])

    total = []
    all = []
    for i in results:
        if "STS" in i and "all" not in i and "SemEval17" not in i:
            total.append(results[i]["pearson"][0])
        if "STS" in i and "all" in i:
            all.append(results[i]["pearson"]["mean"])
    print("Average (datasets): {0}".format(np.mean(total)))
    print("Average (comps): {0}".format(np.mean(all)))