import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The normalized Dirichlet concentration vector describes how evidential mass is distributed "
    "across intents independently of total evidence. OOS queries should often have unusually diffuse "
    "evidence composition relative to in-scope examples predicted as the same intent."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0065"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    a_tr = np.maximum(ctx.h.alpha(tr.logits), eps)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), eps)

    p_tr = a_tr / np.sum(a_tr, axis=1, keepdims=True)
    p_ev = a_ev / np.sum(a_ev, axis=1, keepdims=True)

    k = p_tr.shape[1]
    log_k = np.log(float(max(k, 2)))
    h_tr = -np.sum(p_tr * np.log(p_tr), axis=1) / log_k
    h_ev = -np.sum(p_ev * np.log(p_ev), axis=1) / log_k

    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_mean = float(np.mean(h_tr))
    global_std = float(np.std(h_tr))
    global_std = max(global_std, 1e-4)

    means = np.full(k, global_mean, dtype=float)
    stds = np.full(k, global_std, dtype=float)

    for c in range(k):
        vals = h_tr[pred_tr == c]
        if vals.size >= 3:
            means[c] = float(np.mean(vals))
            stds[c] = max(float(np.std(vals)), 1e-4)

    return (h_ev - means[pred_ev]) / stds[pred_ev]
