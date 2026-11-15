import numpy as np

FAMILY = "logit"
ARM = "edl"
RATIONALE = "In-scope examples have class-dependent Dirichlet concentration patterns: both their total evidence and how that evidence is distributed across intents are typically stable. A class-conditional anomaly score over concentration magnitude and normalized concentration entropy flags atypical evidential shapes without being a monotone transform of energy."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-10

    a_tr = np.maximum(np.asarray(ctx.h.alpha(tr.logits), dtype=float), eps)
    a_ev = np.maximum(np.asarray(ctx.h.alpha(ev.logits), dtype=float), eps)

    def evidence_features(alpha):
        total = np.sum(alpha, axis=1)
        prob = alpha / total[:, None]
        entropy = -np.sum(prob * np.log(np.maximum(prob, eps)), axis=1)
        entropy /= np.log(alpha.shape[1])
        return np.column_stack((np.log(total), entropy))

    x_tr = evidence_features(a_tr)
    x_ev = evidence_features(a_ev)

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_mean = np.mean(x_tr, axis=0)
    centered = x_tr - global_mean
    global_cov = centered.T @ centered / max(len(x_tr) - 1, 1)
    global_cov += 1e-5 * np.eye(2)
    global_inv = np.linalg.pinv(global_cov)

    scores = np.empty(len(x_ev), dtype=float)
    for c in range(k):
        train_mask = pred_tr == c
        eval_mask = pred_ev == c

        if not np.any(eval_mask):
            continue

        xc = x_tr[train_mask]
        if len(xc) >= 4:
            mean = np.mean(xc, axis=0)
            diff = xc - mean
            cov = diff.T @ diff / max(len(xc) - 1, 1)
            cov += 0.05 * global_cov + 1e-5 * np.eye(2)
            inv_cov = np.linalg.pinv(cov)
        else:
            mean = global_mean
            inv_cov = global_inv

        delta = x_ev[eval_mask] - mean
        scores[eval_mask] = np.sum((delta @ inv_cov) * delta, axis=1)

    return np.maximum(scores, 0.0)
