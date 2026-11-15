import numpy as np

FAMILY = "density"
ARM = "ce"
RATIONALE = (
    "A classifier winner that is unusually far under its own shrunk diagonal covariance is suspect, "
    "especially when the runner-up intent has stronger class-conditional density support."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0013", "m_0094"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    x = np.asarray(tr.feats, dtype=float)
    y = np.asarray(tr.labels, dtype=int)
    k = tr.logits.shape[1]
    d = x.shape[1]

    global_mean = np.mean(x, axis=0)
    global_var = np.mean((x - global_mean) ** 2, axis=0)
    global_var = np.maximum(global_var, 1e-8)

    means = np.empty((k, d), dtype=float)
    variances = np.empty((k, d), dtype=float)
    prior_strength = 20.0

    for c in range(k):
        xc = x[y == c]
        n_c = len(xc)
        if n_c == 0:
            means[c] = global_mean
            variances[c] = global_var
        else:
            means[c] = np.mean(xc, axis=0)
            class_var = np.mean((xc - means[c]) ** 2, axis=0)
            shrink = prior_strength / (prior_strength + n_c)
            variances[c] = (1.0 - shrink) * class_var + shrink * global_var

    variances = np.maximum(variances, global_var[None, :] * 1e-4 + 1e-8)

    def diagonal_distance(z, classes):
        delta = z - means[classes]
        return np.mean((delta * delta) / variances[classes], axis=1)

    def top_two(logits):
        order = np.argpartition(logits, -2, axis=1)[:, -2:]
        a = order[:, 0]
        b = order[:, 1]
        swap = logits[np.arange(len(logits)), a] < logits[np.arange(len(logits)), b]
        winner = np.where(swap, b, a)
        runner = np.where(swap, a, b)
        return winner, runner

    tr_winner, tr_runner = top_two(np.asarray(tr.logits))
    tr_primary = diagonal_distance(x, tr_winner)
    tr_runner_dist = diagonal_distance(x, tr_runner)
    tr_conflict = tr_primary - tr_runner_dist

    radial_center = np.median(tr_primary)
    radial_scale = np.median(np.abs(tr_primary - radial_center)) * 1.4826
    conflict_center = np.median(tr_conflict)
    conflict_scale = np.median(np.abs(tr_conflict - conflict_center)) * 1.4826
    radial_scale = max(float(radial_scale), 1e-6)
    conflict_scale = max(float(conflict_scale), 1e-6)

    ev_x = np.asarray(ev.feats, dtype=float)
    ev_winner, ev_runner = top_two(np.asarray(ev.logits))
    ev_primary = diagonal_distance(ev_x, ev_winner)
    ev_runner_dist = diagonal_distance(ev_x, ev_runner)

    radial_surprise = (ev_primary - radial_center) / radial_scale
    density_conflict = (ev_primary - ev_runner_dist - conflict_center) / conflict_scale
    return radial_surprise + 0.5 * np.maximum(density_conflict, 0.0)
