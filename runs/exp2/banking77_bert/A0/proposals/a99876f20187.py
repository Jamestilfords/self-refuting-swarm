import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = (
    "EDL concentration shapes encode which non-predicted intents receive residual evidence. "
    "An OOS query should exhibit an uncommon predicted-intent/runner-up confusion pattern or an atypical runner-up evidence magnitude."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0101"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]

    def evidence_parts(logits, alpha):
        pred = np.argmax(logits, axis=1)
        a = np.nan_to_num(alpha, nan=1.0, posinf=1e12, neginf=1e-12)
        a = np.maximum(a, 1e-12)
        mass = a / np.maximum(a.sum(axis=1, keepdims=True), 1e-12)
        masked = mass.copy()
        masked[np.arange(masked.shape[0]), pred] = -np.inf
        runner = np.argmax(masked, axis=1)
        lead = mass[np.arange(mass.shape[0]), pred]
        second = mass[np.arange(mass.shape[0]), runner]
        magnitude = np.log(np.maximum(second, 1e-12) / np.maximum(lead, 1e-12))
        return pred, runner, magnitude

    tr_pred, tr_runner, tr_mag = evidence_parts(tr.logits, ctx.h.alpha(tr.logits))
    ev_pred, ev_runner, ev_mag = evidence_parts(ev.logits, ctx.h.alpha(ev.logits))

    pair_count = k * k
    tr_pair = tr_pred * k + tr_runner
    counts = np.bincount(tr_pair, minlength=pair_count).astype(float)
    sums = np.bincount(tr_pair, weights=tr_mag, minlength=pair_count)
    sums2 = np.bincount(tr_pair, weights=tr_mag * tr_mag, minlength=pair_count)

    global_mean = float(np.mean(tr_mag))
    global_var = float(np.mean((tr_mag - global_mean) ** 2))
    global_scale = max(np.sqrt(global_var), 1e-3)

    means = np.full(pair_count, global_mean)
    observed = counts > 0
    means[observed] = sums[observed] / counts[observed]

    variances = np.full(pair_count, global_scale * global_scale)
    enough = counts > 1
    variances[enough] = np.maximum(
        sums2[enough] / counts[enough] - means[enough] ** 2,
        (0.05 * global_scale) ** 2,
    )
    scales = np.sqrt(variances)

    # Laplace-smoothed conditional rarity of the runner-up intent given prediction.
    pred_counts = np.bincount(tr_pred, minlength=k).astype(float)
    ev_pair = ev_pred * k + ev_runner
    rarity = -np.log((counts[ev_pair] + 1.0) / (pred_counts[ev_pred] + k))

    magnitude_deviation = np.abs((ev_mag - means[ev_pair]) / scales[ev_pair])
    return np.nan_to_num(rarity + 0.5 * np.minimum(magnitude_deviation, 20.0),
                         nan=0.0, posinf=50.0, neginf=0.0)
