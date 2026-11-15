import numpy as np

FAMILY = "fusion"
ARM = "ce"
RATIONALE = (
    "Shrunk diagonal class-conditional densities capture complementary near-intent deviations "
    "in bottleneck space and broader semantic shifts in pooled encoder space. Train-fitted "
    "minimum-variance weights account for correlation between the two in-scope surprise signals."
)
SOURCE = ""
PARENTS = []
MEMORY_USED = ["m_0069", "m_0295"]


def score(ctx):
    tr, ev = ctx.train, ctx.eval
    k = tr.logits.shape[1]
    ridge = 1e-6
    shrink = 12.0

    def class_nll(train_x, test_x):
        d = train_x.shape[1]
        global_mean = np.mean(train_x, axis=0)
        global_var = np.var(train_x, axis=0) + ridge

        means = np.empty((k, d), dtype=float)
        variances = np.empty((k, d), dtype=float)
        for c in range(k):
            xc = train_x[tr.labels == c]
            n = xc.shape[0]
            if n == 0:
                means[c] = global_mean
                variances[c] = global_var
            else:
                means[c] = np.mean(xc, axis=0)
                local_var = np.var(xc, axis=0)
                variances[c] = (
                    n * local_var + shrink * global_var
                ) / (n + shrink)
                variances[c] = np.maximum(variances[c], ridge)

        def selected_nll(x, pred):
            mu = means[pred]
            va = variances[pred]
            z = (x - mu) ** 2 / va
            return 0.5 * (np.sum(z, axis=1) + np.sum(np.log(va), axis=1))

        train_pred = np.argmax(tr.logits, axis=1)
        test_pred = np.argmax(ev.logits, axis=1)
        return selected_nll(train_x, train_pred), selected_nll(test_x, test_pred)

    tr_b, ev_b = class_nll(tr.feats, ev.feats)
    tr_p, ev_p = class_nll(tr.pooled, ev.pooled)

    train_pred = np.argmax(tr.logits, axis=1)
    eval_pred = np.argmax(ev.logits, axis=1)

    def predicted_class_standardize(train_values, eval_values):
        global_mu = np.mean(train_values)
        global_sd = np.std(train_values) + ridge
        z_train = np.empty_like(train_values, dtype=float)
        z_eval = np.empty_like(eval_values, dtype=float)

        for c in range(k):
            idx = train_pred == c
            n = int(np.sum(idx))
            if n >= 5:
                mu = np.mean(train_values[idx])
                sd = np.std(train_values[idx]) + ridge
            else:
                mu = global_mu
                sd = global_sd
            z_train[idx] = (train_values[idx] - mu) / sd
            z_eval[eval_pred == c] = (eval_values[eval_pred == c] - mu) / sd
        return z_train, z_eval

    zb_tr, zb_ev = predicted_class_standardize(tr_b, ev_b)
    zp_tr, zp_ev = predicted_class_standardize(tr_p, ev_p)

    z_train = np.column_stack((zb_tr, zp_tr))
    cov = np.cov(z_train, rowvar=False)
    cov = np.asarray(cov, dtype=float) + 0.05 * np.eye(2)
    inv_cov = np.linalg.inv(cov)
    ones = np.ones(2)
    weights = inv_cov @ ones
    weights = np.maximum(weights, 0.0)
    if np.sum(weights) <= ridge:
        weights = np.array([0.5, 0.5])
    else:
        weights = weights / np.sum(weights)

    out = weights[0] * zb_ev + weights[1] * zp_ev
    return np.nan_to_num(out, nan=0.0, posinf=1e12, neginf=-1e12)
