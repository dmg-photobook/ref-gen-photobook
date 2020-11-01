"""
BERTScore (Zhang et al. 2019) is an automatic evaluation metric for text generation.

Zhang et al. 2019: https://arxiv.org/abs/1904.09675

"BERTScore computes a similarity score for each token in the candidate sentence with each token in the reference
sentence. However, instead of exact matches, it computes token similarity using contextual embeddings."

"""

import numpy as np


def bert_precision(reference, candidate, stopwords=None):
    """
    Zhang et al. 2019
    """
    if stopwords is None:
        stopwords = set()

    P = 0
    n_tokens = 0
    for w_t in candidate:
        n_tokens += 1
        cosines = []

        for tok_t, v_t in reference:
            if tok_t not in stopwords:
                cosines.append(np.dot(w_t, v_t))

        if cosines:
            P += np.max(cosines)

    # Normalise
    P = P / n_tokens if n_tokens else 0

    return P


def bert_recall(reference, candidate, stopwords=None):
    """
    Zhang et al. 2019
    """
    if stopwords is None:
        stopwords = set()

    R = 0
    n_tokens = 0
    for tok_t, w_t in reference:
        if tok_t in stopwords:
            continue
        else:
            n_tokens += 1

        cosines = [np.dot(w_t, v_t) for v_t in candidate]
        if cosines:
            R += np.max(cosines)

    # Normalise
    R = R / n_tokens if n_tokens else 0

    return R


def bert_f1(reference, candidate, stopwords=None):
    """
    Zhang et al. 2019
    """
    r = bert_recall(reference, candidate, stopwords)
    p = bert_precision(reference, candidate, stopwords)

    if p + r == 0:
        return 0
    else:
        return 2 * ((p * r) / (p + r))


def mean_bert_precision(references, candidate, stopwords=None):
    return np.mean([bert_precision(ref, candidate, stopwords=stopwords) for ref in references])


def mean_bert_recall(references, candidate, stopwords=None):
    return np.mean([bert_recall(ref, candidate, stopwords=stopwords) for ref in references])


def mean_bert_f1(references, candidate, stopwords=None):
    return np.mean([bert_f1(ref, candidate, stopwords=stopwords) for ref in references])