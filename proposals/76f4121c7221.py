import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "OOS queries can have elevated overall logit magnitude while lacking the decisive top-class "
    "separation typical of in-scope predictions. A train-calibrated combination of low top-two "
    "margin and atypical centered logit norm captures this logit-shape mismatch."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def features(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        norm = np.sqrt(np.mean(centered * centered, axis=1))
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return margin, norm

    tr_margin, tr_norm = features(tr.logits)
    ev_margin, ev_norm = features(ev.logits)

    margin_med = np.median(tr_margin)
    margin_scale = np.median(np.abs(tr_margin - margin_med)) + 1e-6
    norm_med = np.median(tr_norm)
    norm_scale = np.median(np.abs(tr_norm - norm_med)) + 1e-6

    low_margin = (margin_med - ev_margin) / margin_scale
    norm_deviation = np.abs(ev_norm - norm_med) / norm_scale
    scores = low_margin + 0.35 * norm_deviation
    return np.asarray(scores, dtype=float)
