import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "Known intents have characteristic ordered winner/runner-up confusion pairs and top-two margins. "
    "A query is OOS-like when its predicted confusion pattern is rare in training or its margin is atypical for that pair."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0091"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    z_tr = np.asarray(tr.logits, dtype=np.float64)
    z_ev = np.asarray(ev.logits, dtype=np.float64)
    k = z_tr.shape[1]

    if k < 2:
        center = np.mean(z_tr, axis=1)
        scale = np.std(center) + 1e-8
        return np.abs((np.mean(z_ev, axis=1) - np.mean(center)) / scale)

    def winner_runner_margin(z):
        winner = np.argmax(z, axis=1)
        top = z[np.arange(z.shape[0]), winner]
        masked = z.copy()
        masked[np.arange(z.shape[0]), winner] = -np.inf
        runner = np.argmax(masked, axis=1)
        second = masked[np.arange(z.shape[0]), runner]
        return winner, runner, top - second

    win_tr, run_tr, margin_tr = winner_runner_margin(z_tr)
    win_ev, run_ev, margin_ev = winner_runner_margin(z_ev)

    pair_tr = win_tr * k + run_tr
    pair_ev = win_ev * k + run_ev
    n_pairs = k * k

    pair_count = np.bincount(pair_tr, minlength=n_pairs).astype(np.float64)
    pair_sum = np.bincount(pair_tr, weights=margin_tr, minlength=n_pairs)
    pair_sq = np.bincount(pair_tr, weights=margin_tr * margin_tr, minlength=n_pairs)

    class_count = np.bincount(win_tr, minlength=k).astype(np.float64)
    class_sum = np.bincount(win_tr, weights=margin_tr, minlength=k)
    class_sq = np.bincount(win_tr, weights=margin_tr * margin_tr, minlength=k)

    global_mean = float(np.mean(margin_tr))
    global_second = float(np.mean(margin_tr * margin_tr))
    class_mean = (class_sum + 5.0 * global_mean) / (class_count + 5.0)
    class_second = (class_sq + 5.0 * global_second) / (class_count + 5.0)

    pair_class = np.repeat(np.arange(k), k)
    shrink = pair_count / (pair_count + 8.0)
    raw_pair_mean = np.divide(
        pair_sum,
        np.maximum(pair_count, 1.0),
        out=np.zeros_like(pair_sum),
        where=pair_count > 0,
    )
    raw_pair_second = np.divide(
        pair_sq,
        np.maximum(pair_count, 1.0),
        out=np.zeros_like(pair_sq),
        where=pair_count > 0,
    )

    pair_mean = shrink * raw_pair_mean + (1.0 - shrink) * class_mean[pair_class]
    pair_second = shrink * raw_pair_second + (1.0 - shrink) * class_second[pair_class]
    pair_var = np.maximum(pair_second - pair_mean * pair_mean, 1e-6)

    # Laplace-smoothed conditional surprise of the runner-up given the winner.
    pair_prob = (pair_count[pair_ev] + 1.0) / (class_count[win_ev] + k)
    rarity = -np.log(np.maximum(pair_prob, 1e-12))

    deviation = (margin_ev - pair_mean[pair_ev]) ** 2 / pair_var[pair_ev]
    scores = rarity + 0.5 * np.minimum(deviation, 25.0)
    return np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
