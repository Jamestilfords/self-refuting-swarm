import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "In-scope predictions tend to have both a clear top-two logit separation and a stronger centered logit magnitude. OOS queries often produce flatter, lower-contrast logit shapes, so train-calibrated deficits in both quantities are combined."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def shape_stats(logits):
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        norm = np.sqrt(np.sum(centered * centered, axis=1))
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        return margin, norm

    tr_margin, tr_norm = shape_stats(np.asarray(tr.logits, dtype=float))
    ev_margin, ev_norm = shape_stats(np.asarray(ev.logits, dtype=float))

    margin_med = np.median(tr_margin)
    norm_med = np.median(tr_norm)
    margin_scale = np.median(np.abs(tr_margin - margin_med)) * 1.4826
    norm_scale = np.median(np.abs(tr_norm - norm_med)) * 1.4826
    margin_scale = max(float(margin_scale), 1e-6)
    norm_scale = max(float(norm_scale), 1e-6)

    margin_deficit = (margin_med - ev_margin) / margin_scale
    norm_deficit = (norm_med - ev_norm) / norm_scale
    scores = 0.6 * margin_deficit + 0.4 * norm_deficit
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
