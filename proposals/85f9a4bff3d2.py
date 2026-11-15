import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "The sorted normalized Dirichlet concentration spectrum has an elbow structure describing "
    "how evidence separates a leading coalition of intents from weaker alternatives. "
    "Class-conditioned novelty of this spectrum can flag unfamiliar evidential competition without using total evidence or vacuity."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0301"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    def elbow_features(logits):
        alpha = np.maximum(ctx.h.alpha(logits), 1e-12)
        p = alpha / np.maximum(alpha.sum(axis=1, keepdims=True), 1e-12)
        lp = np.log(np.maximum(p, 1e-12))
        ordered = np.sort(lp, axis=1)[:, ::-1]
        n, k = ordered.shape

        if k < 2:
            return np.zeros((n, 5), dtype=float)

        gaps = np.maximum(ordered[:, :-1] - ordered[:, 1:], 0.0)
        total = gaps.sum(axis=1, keepdims=True)
        q = gaps / np.maximum(total, 1e-12)

        positions = np.arange(k - 1, dtype=float)
        positions /= max(k - 2, 1)

        entropy = -(q * np.log(np.maximum(q, 1e-12))).sum(axis=1)
        entropy /= np.log(max(k - 1, 2))

        mean_pos = q @ positions
        var_pos = ((positions[None, :] - mean_pos[:, None]) ** 2 * q).sum(axis=1)
        max_gap = q.max(axis=1)
        top_share = np.sort(q, axis=1)[:, -min(3, k - 1):].sum(axis=1)

        return np.column_stack((max_gap, top_share, entropy, mean_pos, var_pos))

    train_x = elbow_features(tr.logits)
    eval_x = elbow_features(ev.logits)
    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    global_mean = train_x.mean(axis=0)
    global_std = train_x.std(axis=0)
    global_std = np.maximum(global_std, 1e-4)

    k_classes = tr.logits.shape[1]
    means = np.tile(global_mean, (k_classes, 1))
    scales = np.tile(global_std, (k_classes, 1))

    for c in range(k_classes):
        mask = train_pred == c
        count = int(mask.sum())
        if count == 0:
            continue

        local = train_x[mask]
        local_mean = local.mean(axis=0)
        local_std = local.std(axis=0)

        # Light shrinkage keeps rarely predicted intents from producing unstable tails.
        weight = count / (count + 8.0)
        means[c] = weight * local_mean + (1.0 - weight) * global_mean
        scales[c] = np.maximum(
            weight * local_std + (1.0 - weight) * global_std,
            1e-4,
        )

    z = (eval_x - means[eval_pred]) / scales[eval_pred]
    scores = np.mean(z * z, axis=1)

    # A negligible continuous tie-breaker preserves ordering for otherwise identical tail scores.
    alpha_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)
    shape_tie = (alpha_ev / alpha_ev.sum(axis=1, keepdims=True)) @ np.arange(alpha_ev.shape[1])
    scores = scores + 1e-10 * shape_tie

    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
