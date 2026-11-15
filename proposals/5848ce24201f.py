import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Maximum-softmax uncertainty and classifier-selected bottleneck deviation expose complementary logit and representation failures. Class-conditional empirical tail calibration learned only from training predictions makes their joint surprise comparable across intents."
SOURCE = ""
PARENTS = ["msp", "d_port"]
MEMORY_USED = ["m_0236", "m_0244"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]
    eps = 1e-12

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    train_uncertainty = 1.0 - ctx.h.msp(tr.logits)
    eval_uncertainty = 1.0 - ctx.h.msp(ev.logits)

    x = np.asarray(tr.feats, dtype=float)
    z = np.asarray(ev.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    dim = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.var(x, axis=0) + 1e-6
    means = np.empty((k, dim), dtype=float)
    variances = np.empty((k, dim), dtype=float)

    for c in range(k):
        xc = x[y == c]
        if len(xc) == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            local_var = np.var(xc, axis=0) if len(xc) > 1 else global_var
            shrink = len(xc) / (len(xc) + 20.0)
            variances[c] = shrink * local_var + (1.0 - shrink) * global_var + 1e-6

    train_delta = x - means[train_pred]
    eval_delta = z - means[eval_pred]
    train_dist = np.mean((train_delta * train_delta) / variances[train_pred], axis=1)
    eval_dist = np.mean((eval_delta * eval_delta) / variances[eval_pred], axis=1)

    out = np.empty(len(eval_pred), dtype=float)
    global_u = np.sort(train_uncertainty)
    global_d = np.sort(train_dist)

    for c in range(k):
        idx = np.where(eval_pred == c)[0]
        if len(idx) == 0:
            continue

        mask = train_pred == c
        ref_u = np.sort(train_uncertainty[mask])
        ref_d = np.sort(train_dist[mask])
        if len(ref_u) < 4:
            ref_u = global_u
            ref_d = global_d

        pu = (np.searchsorted(ref_u, eval_uncertainty[idx], side="right") + 1.0) / (len(ref_u) + 2.0)
        pd = (np.searchsorted(ref_d, eval_dist[idx], side="right") + 1.0) / (len(ref_d) + 2.0)
        out[idx] = -np.log(np.maximum((1.0 - pu) * (1.0 - pd), eps))

    return out
