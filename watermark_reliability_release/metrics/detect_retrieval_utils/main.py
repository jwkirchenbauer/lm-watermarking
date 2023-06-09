import random
import numpy as np
import sys
import argparse
import torch
from models import Averaging, LSTM, load_model
from utils import unk_string
import h5py

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

parser = argparse.ArgumentParser()

parser.add_argument("--data-file", required=True, help="training data")
parser.add_argument("--vocab-file", required=True, help="vocab file")
parser.add_argument("--gpu", default=1, type=int, help="whether to train on gpu")
parser.add_argument("--dim", default=300, type=int, help="dimension of input embeddings")
parser.add_argument("--model", default="avg", choices=["avg", "lstm"], help="type of base model to train.")
parser.add_argument("--grad-clip", default=5., type=float, help='clip threshold of gradients')
parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--dropout", default=0., type=float, help="dropout rate")
parser.add_argument("--batchsize", default=128, type=int, help="size of batches")
parser.add_argument("--megabatch-size", default=60, type=int, help="number of batches in megabatch")
parser.add_argument("--megabatch-anneal", default=150., type=int, help="rate of megabatch annealing in terms of "
                                                                       "number of batches to process before incrementing")
parser.add_argument("--pool", default="mean", choices=["mean", "max"], help="type of pooling")
parser.add_argument("--zero-unk", default=1, type=int, help="whether to ignore unknown tokens")
parser.add_argument("--load-file", help="filename to load a pretrained model.")
parser.add_argument("--save-every-epoch", default=0, type=int, help="whether to save a checkpoint every epoch")
parser.add_argument("--save-final", type=int, default=0, help="save final model")
parser.add_argument("--save-interval", type=int, default=10000, help="frequency (in batches) to evaluate and save model")
parser.add_argument("--report-interval", type=int, default=10000, help="frequency (in batches) to report training status of epoch")
parser.add_argument("--outfile", default="model", help="output file name")
parser.add_argument("--hidden-dim", default=150, type=int, help="hidden dim size of LSTM")
parser.add_argument("--delta", default=0.4, type=float, help="margin")
parser.add_argument("--ngrams", default=0, type=int, help="whether to use character n-grams")
parser.add_argument("--share-encoder", default=1, type=int, help="whether to share the encoder (LSTM only)")
parser.add_argument("--share-vocab", default=1, type=int, help="whether to share the embeddings")
parser.add_argument("--scramble-rate", default=0, type=float, help="rate of scrambling")
parser.add_argument("--sp-model", help="SP model to load for evaluation")
parser.add_argument("--lower-case", type=int, default=1, help="whether to lowercase eval data")
parser.add_argument("--tokenize", type=int, default=0, help="whether to tokenize eval data")
parser.add_argument("--debug", type=int, default=0, help="debug mode")

args = parser.parse_args()

def load_vocab(f):
    f = open(f, 'r')
    lines = f.readlines()
    vocab = {}
    for i in lines:
        i = i.strip()
        i = i.split("\t")
        vocab[i[0]] = int(i[1])
    vocab[unk_string] = len(vocab)
    return vocab

assert args.ngrams == 0
assert args.share_vocab == 1

data = h5py.File(args.data_file, 'r')
data = data['data']
vocab = load_vocab(args.vocab_file)
vocab_fr = None

if args.load_file is not None:
    model, epoch = load_model(data, args)
    print("Loaded model at epoch {0} and resuming training.".format(epoch))
    model.train_epochs(start_epoch=epoch)
else:
    if args.model == "avg":
        model = Averaging(data, args, vocab, vocab_fr)
    elif args.model == "lstm":
        model = LSTM(data, args, vocab, vocab_fr)

    print(" ".join(sys.argv))
    print("Num examples:", len(data))
    print("Num words:", len(vocab))

    model.train_epochs()
