import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "Distance to the classifier-predicted intent mean should be small at a given prediction margin for in-scope queries. "
    "A predicted-class and confidence-conditioned residual tail score flags representations unsupported despite confident classifier decisions."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0032", "m_0121"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    g = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    train_d_all = np.nan_to_num(
        g.class_mahalanobis_sq(tr.pooled),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )
    eval_d_all = np.nan_to_num(
        g.class_mahalanobis_sq(ev.pooled),
        nan=0.0,
        posinf=1e12,
        neginf=0.0,
    )

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)
    n_classes = tr.logits.shape[1]

    if n_classes > 1:
        train_top2 = np.partition(tr.logits, -2, axis=1)[:, -2:]
        eval_top2 = np.partition(ev.logits, -2, axis=1)[:, -2:]
        train_margin = train_top2[:, 1] - train_top2[:, 0]
        eval_margin = eval_top2[:, 1] - eval_top2[:, 0]
    else:
        train_margin = tr.logits[:, 0]
        eval_margin = ev.logits[:, 0]

    train_selected_d = train_d_all[np.arange(len(tr.labels)), train_pred]
    eval_selected_d = eval_d_all[np.arange(len(eval_pred)), eval_pred]

    slopes = np.zeros(n_classes, dtype=float)
    intercepts = np.zeros(n_classes, dtype=float)
    scales = np.ones(n_classes, dtype=float)
    residual_tables = []

    for c in range(n_classes):
        idx = np.flatnonzero(train_pred == c)
        if idx.size == 0:
            idx = np.flatnonzero(tr.labels == c)
        if idx.size == 0:
            idx = np.arange(len(tr.labels))

        x = train_margin[idx]
        y = train_selected_d[idx]
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        denom = float(np.sum((x - x_mean) ** 2))
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / (denom + 1e-8))
        intercept = y_mean - slope * x_mean
        resid = y - (intercept + slope * x)

        scale = float(np.median(np.abs(resid - np.median(resid))) * 1.4826)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = float(np.std(resid) + 1e-6)

        slopes[c] = slope
        intercepts[c] = intercept
        scales[c] = scale
        residual_tables.append(np.sort(resid))

    expected = intercepts[eval_pred] + slopes[eval_pred] * eval_margin
    residual = eval_selected_d - expected

    out = np.empty(len(eval_pred), dtype=float)
    for c in range(n_classes):
        idx = np.flatnonzero(eval_pred == c)
        if idx.size == 0:
            continue
        table = residual_tables[c]
        ranks = np.searchsorted(table, residual[idx], side="right")
        tail = ranks / (len(table) + 1.0)
        detail = np.tanh(residual[idx] / (scales[c] + 1e-8))
        out[idx] = tail + 1e-4 * detail

    return np.nan_to_num(out, nan=0.0, posinf=1.0001, neginf=-0.0001)
