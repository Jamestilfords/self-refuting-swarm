import numpy as np

FAMILY = "knn"
ARM = "edl"
RATIONALE = "The sorted Dirichlet concentration profile captures how evidential mass is distributed across intents, not just its total amount. OOS queries can have atypical evidence-spectrum shapes or precision even when their vacuity or energy resembles in-scope examples."
SOURCE = "https://ar5iv.labs.arxiv.org/html/2203.07798"
PARENTS = []
MEMORY_USED = ["m_0045"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def evidence_shape(logits):
        alpha = np.clip(ctx.h.alpha(logits), 1e-12, 1e12)
        total = alpha.sum(axis=1, keepdims=True)
        p = alpha / total
        k = p.shape[1]

        entropy = -(p * np.log(p)).sum(axis=1) / np.log(max(k, 2))
        ordered = np.sort(p, axis=1)
        top1 = ordered[:, -1]
        top5 = ordered[:, -min(5, k):].sum(axis=1)
        tail = ordered[:, :max(1, k // 2)].sum(axis=1)
        precision = np.log(total[:, 0])

        return np.column_stack((entropy, top1, top5, tail, precision))

    train_x = evidence_shape(tr.logits)
    eval_x = evidence_shape(ev.logits)

    center = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)

    train_x = (train_x - center) / scale
    eval_x = (eval_x - center) / scale

    k = min(25, train_x.shape[0])
    scores = ctx.h.knn(train_x, eval_x, k=k, l2=False)
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
