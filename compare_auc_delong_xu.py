import math
import numpy as np


def _pairwise_vectors(y_true, y_score):
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length")

    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both classes are required for AUC")

    pos_pairs = np.empty((n_pos, n_neg), dtype=float)
    neg_pairs = np.empty((n_neg, n_pos), dtype=float)

    for i, p in enumerate(pos):
        for j, n in enumerate(neg):
            if p > n:
                score = 1.0
            elif p < n:
                score = 0.0
            else:
                score = 0.5
            pos_pairs[i, j] = score

    for j, n in enumerate(neg):
        for i, p in enumerate(pos):
            if p > n:
                score = 1.0
            elif p < n:
                score = 0.0
            else:
                score = 0.5
            neg_pairs[j, i] = score

    v = pos_pairs.mean(axis=1)
    u = neg_pairs.mean(axis=1)
    auc = float(np.mean(v))
    return v, u, auc


def _delong_stats(y_true, y_score):
    v, u, auc = _pairwise_vectors(y_true, y_score)
    var = np.var(v, ddof=1) / len(v) + np.var(u, ddof=1) / len(u)
    return auc, var


def delong_roc_test(y_true, y_pred1, y_pred2):
    y_true = np.asarray(y_true).ravel()
    y_pred1 = np.asarray(y_pred1).ravel()
    y_pred2 = np.asarray(y_pred2).ravel()

    if len(y_true) != len(y_pred1) or len(y_true) != len(y_pred2):
        raise ValueError("y_true, y_pred1 and y_pred2 must have the same length")

    auc1, var1 = _delong_stats(y_true, y_pred1)
    auc2, var2 = _delong_stats(y_true, y_pred2)

    cov = np.cov(v1 := _pairwise_vectors(y_true, y_pred1)[0],
                 v2 := _pairwise_vectors(y_true, y_pred2)[0],
                 bias=False)[0, 1] / len(_pairwise_vectors(y_true, y_pred1)[0])
    cov += np.cov(u1 := _pairwise_vectors(y_true, y_pred1)[1],
                  u2 := _pairwise_vectors(y_true, y_pred2)[1],
                  bias=False)[0, 1] / len(_pairwise_vectors(y_true, y_pred1)[1])

    diff = auc1 - auc2
    var_diff = var1 + var2 - 2 * cov
    if var_diff <= 0:
        return -math.log10(1.0)

    z = diff / math.sqrt(var_diff)
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return math.log10(max(p_value, 1e-300))
