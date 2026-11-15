import numpy as np

FAMILY = "logit"
ARM = "ce"
RATIONALE = (
    "In-scope predictions tend to have characteristic runner-up intents and top-two margins "
    "conditional on the predicted intent. Rare predicted/runner-up pairs or atypical margins "
    "within such pairs indicate unfamiliar classifier competition structure."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0008", "m_0088"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    train_logits = np.nan_to_num(
        np.asarray(tr.logits, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )
    eval_logits = np.nan_to_num(
        np.asarray(ev.logits, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6
    )

    k = train_logits.shape[1]
    tau = 8.0

    def top_pair(logits):
        pred = np.argmax(logits, axis=1)
        masked = logits.copy()
        masked[np.arange(logits.shape[0]), pred] = -np.inf
        runner = np.argmax(masked, axis=1)
        margin = logits[np.arange(logits.shape[0]), pred] - logits[
            np.arange(logits.shape[0]), runner
        ]
        return pred, runner, margin

    pred_tr, run_tr, margin_tr = top_pair(train_logits)
    pred_ev, run_ev, margin_ev = top_pair(eval_logits)

    pair_id = pred_tr * k + run_tr
    pair_count = np.bincount(pair_id, minlength=k * k).astype(float)
    pair_sum = np.bincount(pair_id, weights=margin_tr, minlength=k * k)
    pair_sq = np.bincount(pair_id, weights=margin_tr * margin_tr, minlength=k * k)

    class_count = np.bincount(pred_tr, minlength=k).astype(float)
    class_sum = np.bincount(pred_tr, weights=margin_tr, minlength=k)
    class_sq = np.bincount(pred_tr, weights=margin_tr * margin_tr, minlength=k)

    global_mean = float(np.mean(margin_tr))
    global_var = float(np.var(margin_tr))
    global_var = max(global_var, 1e-8)

    class_mean = np.divide(
        class_sum, class_count, out=np.full(k, global_mean), where=class_count > 0
    )
    class_var = np.divide(
        class_sq, class_count, out=np.full(k, global_var), where=class_count > 0
    ) - class_mean * class_mean
    class_var = np.maximum(class_var, 1e-8)

    pair_mean = np.divide(
        pair_sum, pair_count, out=np.zeros(k * k), where=pair_count > 0
    )
    pair_var = np.divide(
        pair_sq, pair_count, out=np.zeros(k * k), where=pair_count > 0
    ) - pair_mean * pair_mean
    pair_var = np.maximum(pair_var, 0.0)

    ev_pair = pred_ev * k + run_ev
    n = pair_count[ev_pair]
    base_mean = class_mean[pred_ev]
    base_var = class_var[pred_ev]

    shrunk_mean = (n * pair_mean[ev_pair] + tau * base_mean) / (n + tau)
    shrunk_var = (n * pair_var[ev_pair] + tau * base_var) / (n + tau)
    shrunk_var = np.maximum(shrunk_var, 1e-8)

    margin_deviation = np.abs(margin_ev - shrunk_mean) / np.sqrt(shrunk_var)
    rival_rarity = -np.log(
        (n + 1.0) / (class_count[pred_ev] + float(k))
    )

    return np.nan_to_num(
        margin_deviation + 0.35 * rival_rarity,
        nan=0.0,
        posinf=1e6,
        neginf=0.0,
    )
