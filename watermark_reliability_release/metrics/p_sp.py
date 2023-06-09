from argparse import Namespace
import os
import subprocess
import numpy as np
from sacremoses import MosesTokenizer
from metrics.p_sp_utils.models import load_model
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from metrics.p_sp_utils.data_utils import get_df
from metrics.p_sp_utils.evaluate_sts import Example

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

class FileSim(object):

    def __init__(self):
        self.similarity = lambda s1, s2: np.nan_to_num(cosine(np.nan_to_num(s1), np.nan_to_num(s2)))

    def score(self, params, batcher, input1, input2, use_sent_transformers=False):
        sys_scores = []
        if not use_sent_transformers:
            for ii in range(0, len(input1), params.batch_size):
                batch1 = input1[ii:ii + params.batch_size]
                batch2 = input2[ii:ii + params.batch_size]

                # we assume get_batch already throws out the faulty ones
                if len(batch1) == len(batch2) and len(batch1) > 0:
                    enc1 = batcher(params, batch1)
                    enc2 = batcher(params, batch2)

                    for kk in range(enc2.shape[0]):
                        sys_score = self.similarity(enc1[kk], enc2[kk])
                        sys_scores.append(sys_score)
        else:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            #Compute embedding for both lists
            for i in range(len(input1)):
                embedding_1= model.encode(input1[i], convert_to_tensor=True)
                embedding_2 = model.encode(input2[i], convert_to_tensor=True)

                score = util.pytorch_cos_sim(embedding_1, embedding_2)
                sys_scores.append(score.item())
        return sys_scores

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

def evaluate_p_sp(input1, input2, use_sent_transformers=False):
    download_url = 'http://www.cs.cmu.edu/~jwieting/paraphrase-at-scale-english.zip'
    download_dir = './metrics/p_sp_utils'

    args = {
        'gpu': 1 if torch.cuda.is_available() else 0,
        'load_file': './metrics/p_sp_utils/paraphrase-at-scale-english/model.para.lc.100.pt',
        'sp_model': './metrics/p_sp_utils/paraphrase-at-scale-english/paranmt.model',
    }

    # Check if the required files exist
    if not os.path.exists(args['load_file']) or not os.path.exists(args['sp_model']):
        # make a box around the print statement
        print("====================================="*2)
        print("Pretrained model weights wasn't found, Downloading paraphrase-at-scale-english.zip...")
        print("====================================="*2)
        # Download the zip file
        subprocess.run(['wget', download_url])

        # Unzip the file
        subprocess.run(['unzip', 'paraphrase-at-scale-english.zip', '-d', download_dir])

        # Delete the zip file
        os.remove('paraphrase-at-scale-english.zip')

        # Update the file paths
        args['load_file'] = os.path.join(download_dir, 'paraphrase-at-scale-english/model.para.lc.100.pt')
        args['sp_model'] = os.path.join(download_dir, 'paraphrase-at-scale-english/paranmt.model')

    model, _ = load_model(None, args)
    model.eval()

    entok = MosesTokenizer(lang='en')

    new_args = Namespace(batch_size=32, entok=entok, sp=model.sp,
                     params=args, model=model, lower_case=model.args.lower_case,
                     tokenize=model.args.tokenize)
    s = FileSim()
    scores = s.score(new_args, batcher, input1, input2, use_sent_transformers)

    return scores