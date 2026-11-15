import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Known intents should remain compatible with their own coordinate-wise feature variation, "
    "including their most constrained representation directions. A class-specific shrunk diagonal "
    "tail-residual score detects inputs that violate several tight intent-specific constraints."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    xtr = np.asarray(tr.pooled, dtype=float)
    xev = np.asarray(ev.pooled, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    n, d = xtr.shape
    k = tr.logits.shape[1]
    q = max(4, min(32, d // 8))
    eps = 1e-8

    global_var = np.var(xtr, axis=0) + eps
    means = np.zeros((k, d), dtype=float)
    variances = np.zeros((k, d), dtype=float)

    shrink = 20.0
    for c in range(k):
        xc = xtr[y == c]
        if len(xc) == 0:
            means[c] = np.mean(xtr, axis=0)
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            local_var = np.var(xc, axis=0)
            weight = len(xc) / (len(xc) + shrink)
            variances[c] = weight * local_var + (1.0 - weight) * global_var + eps

    def tail_residual(x, classes):
        out = np.empty(len(x), dtype=float)
        for c in range(k):
            idx = np.flatnonzero(classes == c)
            if len(idx) == 0:
                continue
            z2 = (x[idx] - means[c]) ** 2 / variances[c]
            largest = np.partition(z2, d - q, axis=1)[:, d - q:]
            out[idx] = np.mean(largest, axis=1)
        return out

    train_raw = tail_residual(xtr, pred_tr)
    eval_raw = tail_residual(xev, pred_ev)
    scores = np.empty(len(xev), dtype=float)

    for c in range(k):
        idx = np.flatnonzero(pred_ev == c)
        if len(idx) == 0:
            continue
        reference = np.sort(train_raw[pred_tr == c])
        if len(reference) < 8:
            reference = np.sort(train_raw[y == c])
        if len(reference) == 0:
            reference = np.sort(train_raw)
        ranks = np.searchsorted(reference, eval_raw[idx], side="right")
        scores[idx] = ranks / float(len(reference) + 1)

    scale = np.std(train_raw) + eps
    scores += 1e-6 * np.clip((eval_raw - np.mean(train_raw)) / scale, -100.0, 100.0)
    return scores
