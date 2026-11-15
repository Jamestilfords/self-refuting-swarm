import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "The entropy of the non-winning logit allocation detects unusual rival-intent competition, "
    "while predicted-intent diagonal feature distance detects unsupported representation content. "
    "Class-conditional empirical tail mappings learned on training data put both signals on a common scale."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0260", "m_0267"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def rival_entropy(logits):
        p = ctx.h.softmax(logits)
        winner = np.argmax(p, axis=1)
        q = p.copy()
        q[np.arange(q.shape[0]), winner] = 0.0
        mass = np.sum(q, axis=1, keepdims=True)
        q = q / np.maximum(mass, 1e-12)
        ent = -np.sum(q * np.log(np.maximum(q, 1e-12)), axis=1)
        if k > 2:
            ent = ent / np.log(k - 1.0)
        return ent

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    train_logit = rival_entropy(tr.logits)
    eval_logit = rival_entropy(ev.logits)

    x_train = tr.pooled
    x_eval = ev.pooled
    global_mean = np.mean(x_train, axis=0)
    global_var = np.var(x_train, axis=0) + 1e-8

    train_dist = np.empty(x_train.shape[0], dtype=float)
    eval_dist = np.empty(x_eval.shape[0], dtype=float)

    for c in range(k):
        train_idx = np.where(train_pred == c)[0]
        eval_idx = np.where(eval_pred == c)[0]

        if train_idx.size >= 5:
            xc = x_train[train_idx]
            mean = np.mean(xc, axis=0)
            var = np.var(xc, axis=0)
            var = var + 0.05 * global_var + 1e-8
        else:
            mean = global_mean
            var = global_var

        if train_idx.size:
            z = (x_train[train_idx] - mean) / np.sqrt(var)
            train_dist[train_idx] = np.mean(z * z, axis=1)
        if eval_idx.size:
            z = (x_eval[eval_idx] - mean) / np.sqrt(var)
            eval_dist[eval_idx] = np.mean(z * z, axis=1)

    global_logit_sorted = np.sort(train_logit)
    global_dist_sorted = np.sort(train_dist)
    logit_tail = np.empty(ev.logits.shape[0], dtype=float)
    dist_tail = np.empty(ev.logits.shape[0], dtype=float)

    for c in range(k):
        train_idx = np.where(train_pred == c)[0]
        eval_idx = np.where(eval_pred == c)[0]
        if not eval_idx.size:
            continue

        if train_idx.size >= 5:
            ls = np.sort(train_logit[train_idx])
            ds = np.sort(train_dist[train_idx])
        else:
            ls = global_logit_sorted
            ds = global_dist_sorted

        logit_tail[eval_idx] = np.searchsorted(ls, eval_logit[eval_idx], side="right") / float(ls.size)
        dist_tail[eval_idx] = np.searchsorted(ds, eval_dist[eval_idx], side="right") / float(ds.size)

    logit_tail = np.minimum(logit_tail, 1.0 - 1e-8)
    dist_tail = np.minimum(dist_tail, 1.0 - 1e-8)
    return -np.log1p(-logit_tail) - np.log1p(-dist_tail)
