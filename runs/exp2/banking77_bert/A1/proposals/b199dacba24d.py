import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "The joint shape of a classifier's top-two margin and centered logit norm is intent-specific on "
    "training data. OOS inputs can have an atypical confidence magnitude or margin for their predicted intent, "
    "including overconfident extrapolations that energy or MSP may not flag."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def signature(logits):
        x = np.asarray(logits, dtype=float)
        top2 = np.partition(x, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = x - np.mean(x, axis=1, keepdims=True)
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        return np.column_stack((margin, np.log1p(norm)))

    tr_sig = signature(tr.logits)
    ev_sig = signature(ev.logits)
    tr_pred = np.argmax(tr.logits, axis=1)
    ev_pred = np.argmax(ev.logits, axis=1)
    k = tr.logits.shape[1]

    global_med = np.median(tr_sig, axis=0)
    global_mad = np.median(np.abs(tr_sig - global_med), axis=0)
    global_scale = np.maximum(1.4826 * global_mad, 1e-4)

    med = np.tile(global_med, (k, 1))
    scale = np.tile(global_scale, (k, 1))

    for c in range(k):
        rows = tr_sig[tr_pred == c]
        if rows.shape[0] >= 8:
            m = np.median(rows, axis=0)
            s = 1.4826 * np.median(np.abs(rows - m), axis=0)
            med[c] = m
            scale[c] = np.maximum(s, 0.15 * global_scale)

    z = (ev_sig - med[ev_pred]) / scale[ev_pred]
    z = np.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

    # Squared robust deviations retain both unusually low and unusually high confidence shapes.
    return np.minimum(np.sum(z * z, axis=1), 1e8)
