import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two logit margin and centered logit norm characterize confidence geometry beyond energy or maximum probability. Per-predicted-intent training calibration flags queries whose margin-norm relationship is atypical for that intent."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0137"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def features(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return np.column_stack((margin, np.log1p(norm)))

    z_tr = features(tr.logits)
    z_ev = features(ev.logits)
    pred_tr = np.argmax(tr.logits, axis=1)
    pred_ev = np.argmax(ev.logits, axis=1)

    global_mean = np.mean(z_tr, axis=0)
    global_var = np.var(z_tr, axis=0) + 1e-6
    scores = np.empty(z_ev.shape[0], dtype=float)

    for c in range(tr.logits.shape[1]):
        mask_tr = pred_tr == c
        mask_ev = pred_ev == c
        if not np.any(mask_ev):
            continue
        if np.sum(mask_tr) >= 4:
            x = z_tr[mask_tr]
            mean = np.mean(x, axis=0)
            cov = np.cov(x, rowvar=False) + np.eye(2) * 1e-5
            inv_cov = np.linalg.pinv(cov)
            diff = z_ev[mask_ev] - mean
            scores[mask_ev] = np.sum((diff @ inv_cov) * diff, axis=1)
        else:
            diff = z_ev[mask_ev] - global_mean
            scores[mask_ev] = np.sum(diff * diff / global_var, axis=1)

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
