import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Trailing within-class covariance directions encode tightly constrained semantic variation, "
    "so unusually large low-variance residuals indicate unsupported intent content."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0113"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    part = ctx.h.partial(tr.feats, tr.labels, top_k=128)

    train_score = np.asarray(part.bottom(tr.feats), dtype=float).reshape(-1)
    eval_score = np.asarray(part.bottom(ev.feats), dtype=float).reshape(-1)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    global_ref = np.sort(train_score)

    out = np.empty(eval_score.shape[0], dtype=float)
    for cls in range(tr.logits.shape[1]):
        mask = train_pred == cls
        ref = np.sort(train_score[mask]) if np.any(mask) else global_ref
        idx = np.where(eval_pred == cls)[0]
        if idx.size:
            out[idx] = np.searchsorted(ref, eval_score[idx], side="right") / (ref.size + 1.0)

    return out + 1e-8 * np.asarray(ctx.h.energy(ev.logits), dtype=float)
