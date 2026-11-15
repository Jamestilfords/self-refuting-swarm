import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The entropy of the normalized Dirichlet concentration vector measures how evidence is allocated across intents, "
    "and conditioning that shape statistic on total evidence isolates atypical evidential allocation rather than vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0119"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    eps = 1e-12

    def evidence_shape(logits):
        alpha = np.maximum(ctx.h.alpha(logits), eps)
        total = np.sum(alpha, axis=1)
        p = alpha / total[:, None]
        entropy = -np.sum(p * np.log(np.maximum(p, eps)), axis=1)
        return np.log(total), entropy

    t_train, y_train = evidence_shape(tr.logits)
    t_eval, y_eval = evidence_shape(ev.logits)

    n_classes = tr.logits.shape[1]
    global_t_mean = np.mean(t_train)
    global_y_mean = np.mean(y_train)
    global_t_var = np.mean((t_train - global_t_mean) ** 2) + 1e-6
    global_slope = np.mean((t_train - global_t_mean) * (y_train - global_y_mean)) / global_t_var
    global_intercept = global_y_mean - global_slope * global_t_mean
    global_resid = y_train - (global_intercept + global_slope * t_train)
    global_scale = np.sqrt(np.mean(global_resid ** 2) + 1e-6)

    intercept = np.full(n_classes, global_intercept, dtype=float)
    slope = np.full(n_classes, global_slope, dtype=float)
    scale = np.full(n_classes, global_scale, dtype=float)

    for c in range(n_classes):
        mask = tr.labels == c
        count = int(np.sum(mask))
        if count < 3:
            continue
        tc = t_train[mask]
        yc = y_train[mask]
        tm = np.mean(tc)
        ym = np.mean(yc)
        # Mild variance shrinkage prevents unstable fits for small intents.
        var = np.mean((tc - tm) ** 2)
        cov = np.mean((tc - tm) * (yc - ym))
        shrink = count / (count + 10.0)
        b = cov / (var + (1.0 - shrink) * global_t_var + 1e-6)
        a = ym - b * tm
        resid = yc - (a + b * tc)
        local_scale = np.sqrt(np.mean(resid ** 2) + 1e-6)
        intercept[c] = shrink * a + (1.0 - shrink) * global_intercept
        slope[c] = shrink * b + (1.0 - shrink) * global_slope
        scale[c] = shrink * local_scale + (1.0 - shrink) * global_scale

    predicted = np.argmax(ev.logits, axis=1)
    expected_entropy = intercept[predicted] + slope[predicted] * t_eval
    scores = np.abs(y_eval - expected_entropy) / np.maximum(scale[predicted], 1e-6)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
