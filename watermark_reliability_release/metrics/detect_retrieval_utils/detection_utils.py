import json
import numpy as np
import hashlib
import os
import functools
import sys
import numpy as np
import openai
import re
import string
import collections as cll
import time
import requests
import pickle
import torch
import pandas as pd
from transformers import LogitsWarper
from sklearn.metrics import roc_curve, auc


def print_tpr_target(fpr, tpr, name, target_fpr):
    indices = None
    for i in range(len(fpr)):
        if fpr[i] >= target_fpr:
            if i == 0:
                indices = [i]
            else:
                indices = [i-1, i]
            break

    if indices is None:
        print(f"{name} TPR at {target_fpr*100}% FPR: {tpr[-1]}. FPR is too high.")
    else:
        tpr_values = [tpr[i] for i in indices]
        print(f"{name} TPR at {target_fpr*100}% FPR: {np.mean(tpr_values) * 100:5.1f}%")

def get_roc(human_scores, machine_scores, max_fpr=1.0):
    fpr, tpr, _ = roc_curve([0] * len(human_scores) + [1] * len(machine_scores), human_scores + machine_scores)
    fpr_auc = [x for x in fpr if x <= max_fpr]
    tpr_auc = tpr[:len(fpr_auc)]
    roc_auc = auc(fpr_auc, tpr_auc)
    return fpr.tolist(), tpr.tolist(), float(roc_auc), float(roc_auc) * (1.0 / max_fpr)


def f1_score(prediction, ground_truth, gram=1, stopwords=None):
    """Calculate word level F1 score."""
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    prediction_tokens = prediction.split()
    ground_truth_tokens = ground_truth.split()
    prediction_tokens = [
        " ".join(prediction_tokens[i:i + gram])
        for i in range(0, len(prediction_tokens) - gram + 1)
    ]
    ground_truth_tokens = [
        " ".join(ground_truth_tokens[i:i + gram])
        for i in range(0, len(ground_truth_tokens) - gram + 1)
    ]

    if stopwords:
        prediction_tokens = [x for x in prediction_tokens if x not in stopwords]
        ground_truth_tokens = [x for x in ground_truth_tokens if x not in stopwords]

    if not prediction_tokens or not ground_truth_tokens:
        return 1.0, 1.0, 1.0, True
    common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0, False
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1, False

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
