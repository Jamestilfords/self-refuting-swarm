import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope predictions tend to have both a clearer top-two class separation and a larger centered-logit magnitude. "
    "Combining train-calibrated deficits in these two logit-shape measures flags uncertain, low-concentration OOS outputs."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr_logits = np.asarray(ctx.train.logits, dtype=float)
    ev_logits = np.asarray(ctx.eval.logits, dtype=float)

    def shape_values(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        centered = logits - np.mean(logits, axis=1, keepdims=True)
        concentration = np.sqrt(np.mean(centered * centered, axis=1))
        return margin, concentration

    tr_margin, tr_concentration = shape_values(tr_logits)
    ev_margin, ev_concentration = shape_values(ev_logits)

    margin_med = np.median(tr_margin)
    conc_med = np.median(tr_concentration)
    margin_scale = np.median(np.abs(tr_margin - margin_med))
    conc_scale = np.median(np.abs(tr_concentration - conc_med))

    margin_scale = max(float(margin_scale), 1e-6)
    conc_scale = max(float(conc_scale), 1e-6)

    margin_deficit = (margin_med - ev_margin) / margin_scale
    concentration_deficit = (conc_med - ev_concentration) / conc_scale

    return np.nan_to_num(
        0.5 * margin_deficit + 0.5 * concentration_deficit,
        nan=0.0,
        posinf=1e6,
        neginf=-1e6,
    )
