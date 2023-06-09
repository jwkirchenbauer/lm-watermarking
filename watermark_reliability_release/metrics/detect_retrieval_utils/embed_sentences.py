import argparse
import numpy as np
from sacremoses import MosesTokenizer
from metrics.detect_retrieval_utils.models import load_model
from metrics.detect_retrieval_utils.utils import Example
import torch
from argparse import Namespace
import tqdm


entok = MosesTokenizer(lang='en')


def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def similarity(s1, s2):
    return cosine(np.nan_to_num(s1), np.nan_to_num(s2))

def embed(params, sentences):
    results = []
    for ii in range(0, len(sentences), params.batch_size):
        batch1 = sentences[ii:ii + params.batch_size]
        results.extend(batcher(params, batch1))
    return np.vstack(results)

def batcher(params, batch):
    new_batch = []
    for p in batch:
        if params.tokenize:
            tok = params.entok.tokenize(p, escape=False)
            p = " ".join(tok)
        if params.lower_case:
            p = p.lower()
        p = params.sp.EncodeAsPieces(p)
        p = " ".join(p)
        p = Example(p, params.lower_case)
        p.populate_embeddings(params.model.vocab, params.model.zero_unk, params.model.ngrams)
        new_batch.append(p)
    x, l = params.model.torchify_batch(new_batch)
    vecs = params.model.encode(x, l)
    return vecs.detach().cpu().numpy()

def embed_all(model, sentences, disable=False):
    model.eval()
    new_args = Namespace(batch_size=32, entok=entok, sp=model.sp,
                         model=model, lower_case=model.args.lower_case,
                         tokenize=model.args.tokenize)
    all_vecs = []
    for i in tqdm.tqdm(range(0, len(sentences), 10000), disable=disable):
        sents = sentences[i:i + 10000]
        with torch.inference_mode():
            vecs = embed(new_args, sents)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--load-file", help="path to saved model")
    parser.add_argument("--sp-model", help="sentencepiece model to use")
    parser.add_argument("--gpu", default=1, type=int, help="whether to train on gpu")
    parser.add_argument("--sentence-file", help="sentence file")
    parser.add_argument("--output-file", help="prefix for output numpy file")
    args = parser.parse_args()

    model, _ = load_model(args.load_file, args.gpu, args.sp_model)
    print(model.args)
    model.eval()
    embed_all(args, model)
