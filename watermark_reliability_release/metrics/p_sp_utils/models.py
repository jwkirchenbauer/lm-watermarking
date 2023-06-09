import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
import sentencepiece as spm
# import pairing
from metrics.p_sp_utils import pairing
# import utils
from torch.nn.modules.distance import CosineSimilarity
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from .evaluate_sts import evaluate_sts
from torch import optim


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

def load_model(data, args):
    if not args['gpu']:
        model = torch.load(args['load_file'], map_location=torch.device('cpu'))
    else:
        model = torch.load(args['load_file'])

    state_dict = model['state_dict']
    model_args = model['args']
    vocab = model['vocab']
    vocab_fr = model['vocab_fr']
    optimizer = model['optimizer']
    epoch = model['epoch'] + 1

    if 'sp_model' in args:
        model_args.sp_model = args['sp_model']
    if 'megabatch_anneal' in args:
        model_args.megabatch_anneal = args['megabatch_anneal']
    model_args.gpu = args['gpu']

    if model_args.model == "avg":
        model = Averaging(data, model_args, vocab, vocab_fr)
    elif args.model == "lstm":
        model = LSTM(data, model_args, vocab, vocab_fr)

    model.load_state_dict(state_dict)
    model.optimizer.load_state_dict(optimizer)

    return model, epoch

class ParaModel(nn.Module):
    def __init__(self, data, args, vocab, vocab_fr):
        super(ParaModel, self).__init__()

        self.data = data
        self.args = args
        self.gpu = args.gpu
        self.save_interval = args.save_interval
        if "report_interval" in args:
            self.report_interval = args.report_interval
        else:
            self.report_interval = args.save_interval

        self.vocab = vocab
        self.rev_vocab = {v:k for k,v in vocab.items()}
        self.vocab_fr = vocab_fr
        self.ngrams = args.ngrams

        self.delta = args.delta
        self.pool = args.pool

        self.dropout = args.dropout
        self.share_encoder = args.share_encoder
        self.share_vocab = args.share_vocab
        self.scramble_rate = args.scramble_rate
        self.zero_unk = args.zero_unk

        self.batchsize = args.batchsize
        self.max_megabatch_size = args.megabatch_size
        self.curr_megabatch_size = 1
        self.megabatch = []
        self.megabatch_anneal = args.megabatch_anneal
        self.increment = False

        self.sim_loss = nn.MarginRankingLoss(margin=self.delta)
        self.cosine = CosineSimilarity()

        self.embedding = nn.Embedding(len(self.vocab), self.args.dim)
        if self.vocab_fr is not None:
            self.embedding_fr = nn.Embedding(len(self.vocab_fr), self.args.dim)

        self.sp = None
        if args.sp_model:
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(args.sp_model)

    def save_params(self, epoch, counter=None):
        if counter is None:
            torch.save({'state_dict': self.state_dict(),
                    'vocab': self.vocab,
                    'vocab_fr': self.vocab_fr,
                    'args': self.args,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch}, "{0}_{1}.pt".format(self.args.outfile, epoch))
        else:
            torch.save({'state_dict': self.state_dict(),
                    'vocab': self.vocab,
                    'vocab_fr': self.vocab_fr,
                    'args': self.args,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': epoch, 'counter': counter}, "{0}_{1}_{2}.pt".format(self.args.outfile, epoch, counter))

    def torchify_batch(self, batch):
        max_len = 0
        for i in batch:
            if len(i.embeddings) > max_len:
                max_len = len(i.embeddings)

        batch_len = len(batch)

        np_sents = np.zeros((batch_len, max_len), dtype='int32')
        np_lens = np.zeros((batch_len,), dtype='int32')

        for i, ex in enumerate(batch):
            np_sents[i, :len(ex.embeddings)] = ex.embeddings
            np_lens[i] = len(ex.embeddings)

        idxs, lengths = torch.from_numpy(np_sents).long(), \
                               torch.from_numpy(np_lens).float().long()

        if self.gpu:
            idxs = idxs.cuda()
            lengths = lengths.cuda()
    
        return idxs, lengths

    def loss_function(self, g1, g2, p1, p2):
        g1g2 = self.cosine(g1, g2)
        g1p1 = self.cosine(g1, p1)
        g2p2 = self.cosine(g2, p2)

        ones = torch.ones(g1g2.size()[0])
        if self.gpu:
            ones = ones.cuda()

        loss = self.sim_loss(g1g2, g1p1, ones) + self.sim_loss(g1g2, g2p2, ones)

        return loss

    def scoring_function(self, g_idxs1, g_lengths1, g_idxs2, g_lengths2, fr0=0, fr1=0):
        g1 = self.encode(g_idxs1, g_lengths1, fr=fr0)
        g2 = self.encode(g_idxs2, g_lengths2, fr=fr1)
        return self.cosine(g1, g2)

    def train_epochs(self, start_epoch=1):
        start_time = time.time()
        self.megabatch = []
        self.ep_loss = 0
        self.curr_idx = 0

        self.eval()
        evaluate_sts(self, self.args)
        self.train()

        try:
            for ep in range(start_epoch, self.args.epochs+1):
                self.mb = get_minibatches_idx(len(self.data), self.args.batchsize, shuffle=True)
                self.curr_idx = 0
                self.ep_loss = 0
                self.megabatch = []
                cost = 0
                counter = 0

                while(cost is not None):
                    cost = pairing.compute_loss_one_batch(self)
                    if cost is None:
                        continue

                    self.ep_loss += cost.item()
                    counter += 1
                    if counter % self.report_interval == 0:
                        print("Epoch {0}, Counter {1}/{2}".format(ep, counter, len(self.mb)))
                    if self.save_interval > 0 and counter > 0:
                        if counter % self.save_interval == 0:
                            self.eval()
                            evaluate_sts(self, self.args)
                            self.train()
                            self.save_params(ep, counter=counter)

                    self.optimizer.zero_grad()
                    cost.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters, self.args.grad_clip)
                    self.optimizer.step()

                self.eval()
                evaluate_sts(self, self.args)
                self.train()

                if self.args.save_every_epoch:
                    self.save_params(ep)

                print('Epoch {0}\tCost: '.format(ep), self.ep_loss / counter)

        except KeyboardInterrupt:
            print("Training Interrupted")

        if self.args.save_final:
            self.save_params(ep)

        end_time = time.time()
        print("Total Time:", (end_time - start_time))

class Averaging(ParaModel):
    def __init__(self, data, args, vocab, vocab_fr):
        super(Averaging, self).__init__(data, args, vocab, vocab_fr)
        self.parameters = self.parameters()
        self.optimizer = optim.Adam(self.parameters, lr=self.args.lr)

        if args.gpu:
           self.cuda()

        print(self)
        
    def forward(self, curr_batch):
        g_idxs1 = curr_batch.g1
        g_lengths1 = curr_batch.g1_l

        g_idxs2 = curr_batch.g2
        g_lengths2 = curr_batch.g2_l

        p_idxs1 = curr_batch.p1
        p_lengths1 = curr_batch.p1_l

        p_idxs2 = curr_batch.p2
        p_lengths2 = curr_batch.p2_l

        g1 = self.encode(g_idxs1, g_lengths1)
        g2 = self.encode(g_idxs2, g_lengths2, fr=1)
        p1 = self.encode(p_idxs1, p_lengths1, fr=1)
        p2 = self.encode(p_idxs2, p_lengths2)

        return g1, g2, p1, p2

    def encode(self, idxs, lengths, fr=0):
        if fr and not self.share_vocab:
            word_embs = self.embedding_fr(idxs)
        else:
            word_embs = self.embedding(idxs)

        if self.dropout > 0:
            word_embs = F.dropout(word_embs, p=self.dropout, training=self.training)

        if self.pool == "max":
            word_embs = max_pool(word_embs, lengths, self.args.gpu)
        elif self.pool == "mean":
            word_embs = mean_pool(word_embs, lengths, self.args.gpu)

        return word_embs

class LSTM(ParaModel):
    def __init__(self, data, args, vocab, vocab_fr):
        super(LSTM, self).__init__(data, args, vocab, vocab_fr)

        self.hidden_dim = self.args.hidden_dim

        self.e_hidden_init = torch.zeros(2, 1, self.args.hidden_dim)
        self.e_cell_init = torch.zeros(2, 1, self.args.hidden_dim)

        if self.gpu:
            self.e_hidden_init = self.e_hidden_init.cuda()
            self.e_cell_init = self.e_cell_init.cuda()

        self.lstm = nn.LSTM(self.args.dim, self.hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        if not self.share_encoder:
            self.lstm_fr = nn.LSTM(self.args.dim, self.hidden_dim, num_layers=1,
                                       bidirectional=True, batch_first=True)

        self.parameters = self.parameters()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters), self.args.lr)

        if self.gpu:
           self.cuda()

        print(self)

    def encode(self, inputs, lengths, fr=0):
        bsz, max_len = inputs.size()
        e_hidden_init = self.e_hidden_init.expand(2, bsz, self.hidden_dim).contiguous()
        e_cell_init = self.e_cell_init.expand(2, bsz, self.hidden_dim).contiguous()
        lens, indices = torch.sort(lengths, 0, True)

        if fr and not self.share_vocab:
            in_embs = self.embedding_fr(inputs)
        else:
            in_embs = self.embedding(inputs)

        if fr and not self.share_encoder:
            if self.dropout > 0:
                in_embs = F.dropout(in_embs, p=self.dropout, training=self.training)
            all_hids, (enc_last_hid, _) = self.lstm_fr(pack(in_embs[indices],
                                                        lens.tolist(), batch_first=True), (e_hidden_init, e_cell_init))
        else:
            if self.dropout > 0:
                in_embs = F.dropout(in_embs, p=self.dropout, training=self.training)
            all_hids, (enc_last_hid, _) = self.lstm(pack(in_embs[indices],
                                                         lens.tolist(), batch_first=True), (e_hidden_init, e_cell_init))

        _, _indices = torch.sort(indices, 0)
        all_hids = unpack(all_hids, batch_first=True)[0][_indices]

        if self.pool == "max":
            embs = max_pool(all_hids, lengths, self.gpu)
        elif self.pool == "mean":
            embs = mean_pool(all_hids, lengths, self.gpu)
        return embs

    def forward(self, curr_batch):
        g_idxs1 = curr_batch.g1
        g_lengths1 = curr_batch.g1_l

        g_idxs2 = curr_batch.g2
        g_lengths2 = curr_batch.g2_l

        p_idxs1 = curr_batch.p1
        p_lengths1 = curr_batch.p1_l

        p_idxs2 = curr_batch.p2
        p_lengths2 = curr_batch.p2_l

        g1 = self.encode(g_idxs1, g_lengths1)
        g2 = self.encode(g_idxs2, g_lengths2, fr=1)
        p1 = self.encode(p_idxs1, p_lengths1, fr=1)
        p2 = self.encode(p_idxs2, p_lengths2)

        return g1, g2, p1, p2
