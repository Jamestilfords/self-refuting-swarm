import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Trailing within-class covariance directions encode tight, low-variance constraints of each known intent. "
    "Calibrating their residual separately for the classifier-predicted intent prevents naturally variable intents from dominating the OOS ranking."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    detector = ctx.h.partial(tr.pooled, tr.labels, top_k=128)
    train_raw = np.asarray(detector.bottom(tr.pooled), dtype=float).reshape(-1)
    eval_raw = np.asarray(detector.bottom(ev.pooled), dtype=float).reshape(-1)

    finite_train = train_raw[np.isfinite(train_raw)]
    if finite_train.size == 0:
        finite_train = np.array([0.0])
    global_center = np.median(finite_train)
    global_scale = np.median(np.abs(finite_train - global_center)) + 1e-8

    train_clean = np.nan_to_num(
        train_raw, nan=global_center,
        posinf=global_center + 100.0 * global_scale,
        neginf=global_center - 100.0 * global_scale
    )
    eval_clean = np.nan_to_num(
        eval_raw, nan=global_center,
        posinf=global_center + 100.0 * global_scale,
        neginf=global_center - 100.0 * global_scale
    )

    pred = np.argmax(ev.logits, axis=1)
    labels = np.asarray(tr.labels, dtype=int)
    sorted_by_class = {}
    for cls in np.unique(labels):
        values = np.sort(train_clean[labels == cls])
        if values.size:
            sorted_by_class[int(cls)] = values

    out = np.empty(eval_clean.shape[0], dtype=float)
    global_sorted = np.sort(train_clean)
    for cls in np.unique(pred):
        mask = pred == cls
        reference = sorted_by_class.get(int(cls), global_sorted)
        ranks = np.searchsorted(reference, eval_clean[mask], side="right")
        out[mask] = ranks / (reference.size + 1.0)

    residual = np.tanh((eval_clean - global_center) / global_scale)
    return np.nan_to_num(out + 1e-4 * residual, nan=0.0, posinf=1.0, neginf=0.0)
