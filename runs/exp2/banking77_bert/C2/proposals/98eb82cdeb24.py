import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = "Bottleneck and encoder Gaussian densities capture complementary local intent deviations and broader domain shifts. Their additive weights are fitted on training examples to predict training classification loss, emphasizing the density signal most associated with uncertain in-scope behavior."
SOURCE = ""
PARENTS = ["d_port", "d_full"]
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    g_port = ctx.h.gda(tr.feats, tr.labels, cov="tied")
    g_full = ctx.h.gda(tr.pooled, tr.labels, cov="tied")

    port_train = g_port.neg_log_density(tr.feats)
    full_train = g_full.neg_log_density(tr.pooled)
    port_eval = g_port.neg_log_density(ev.feats)
    full_eval = g_full.neg_log_density(ev.pooled)

    def standardize(train_values, eval_values):
        center = np.median(train_values)
        scale = np.std(train_values)
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        return (train_values - center) / scale, (eval_values - center) / scale

    z_port_train, z_port_eval = standardize(port_train, port_eval)
    z_full_train, z_full_eval = standardize(full_train, full_eval)

    probs = ctx.h.softmax(tr.logits)
    true_prob = probs[np.arange(len(tr.labels)), tr.labels]
    loss = -np.log(np.clip(true_prob, 1e-12, 1.0))
    loss = loss - np.mean(loss)

    design = np.column_stack([z_port_train, z_full_train])
    gram = design.T @ design
    rhs = design.T @ loss
    weights = np.linalg.solve(gram + 1e-3 * np.eye(2), rhs)
    weights = np.maximum(weights, 0.0)

    if not np.all(np.isfinite(weights)) or np.sum(weights) < 1e-10:
        weights = np.array([0.35, 0.65])
    else:
        weights = weights / np.sum(weights)

    scores = weights[0] * z_port_eval + weights[1] * z_full_eval
    return np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=-1e12)
