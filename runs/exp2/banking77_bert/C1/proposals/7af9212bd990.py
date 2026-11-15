FAMILY = "density"
ARM = "ce"
RATIONALE = "Low-variance residuals are most informative when calibrated against the classifier's predicted intent, since different intents can have different within-class trailing-direction spread. This converts the trailing eigenspace residual into a predicted-class conditional atypicality score."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    import numpy as np

    tr, ev = ctx.train, ctx.eval
    partial = ctx.h.partial(tr.pooled, tr.labels, top_k=128)

    train_resid = np.asarray(partial.bottom(tr.pooled), dtype=float)
    eval_resid = np.asarray(partial.bottom(ev.pooled), dtype=float)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_med = float(np.median(train_resid))
    global_scale = float(
        np.quantile(train_resid, 0.75) - np.quantile(train_resid, 0.25)
    )
    global_scale = max(global_scale, 1e-8)

    med = np.full(k, global_med, dtype=float)
    scale = np.full(k, global_scale, dtype=float)

    for c in range(k):
        values = train_resid[train_pred == c]
        if values.size >= 8:
            med[c] = np.median(values)
            spread = np.quantile(values, 0.75) - np.quantile(values, 0.25)
            scale[c] = max(float(spread), 1e-8)

    scores = (eval_resid - med[eval_pred]) / scale[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
