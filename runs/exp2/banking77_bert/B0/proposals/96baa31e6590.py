import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = "The top-two logit margin measures local intent ambiguity that energy and maximum softmax probability can miss. Standardizing the margin with train-only class-conditional statistics accounts for different intrinsic confidence scales across predicted intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0014"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def margin_and_pred(logits):
        top2 = np.partition(logits, -2, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        pred = np.argmax(logits, axis=1)
        return margin, pred

    train_margin, train_pred = margin_and_pred(tr.logits)
    eval_margin, eval_pred = margin_and_pred(ev.logits)
    n_classes = tr.logits.shape[1]

    sums = np.bincount(train_pred, weights=train_margin, minlength=n_classes)
    counts = np.bincount(train_pred, minlength=n_classes).astype(float)
    global_mean = float(np.mean(train_margin))
    means = (sums + 5.0 * global_mean) / (counts + 5.0)

    centered = train_margin - means[train_pred]
    sq_sums = np.bincount(
        train_pred, weights=centered * centered, minlength=n_classes
    )
    global_scale = float(np.sqrt(np.mean(centered * centered) + 1e-8))
    scales = np.sqrt(
        (sq_sums + 5.0 * global_scale * global_scale) / (counts + 5.0) + 1e-8
    )

    scores = -(eval_margin - means[eval_pred]) / scales[eval_pred]
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
