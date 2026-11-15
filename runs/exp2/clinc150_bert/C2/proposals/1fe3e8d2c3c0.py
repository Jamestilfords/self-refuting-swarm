import numpy as np

FAMILY = "other"
ARM = "edl"
RATIONALE = "This measures how diffuse the normalized Dirichlet concentration vector is after accounting for total evidence strength learned on training data. OOS inputs can retain substantial evidence while distributing it unusually evenly across intents."
SOURCE = ""
PARENTS = []
MEMORY_USED = []


def score(ctx):
    tr, ev = ctx.train, ctx.eval

    a_tr = np.maximum(ctx.h.alpha(tr.logits), 1e-12)
    a_ev = np.maximum(ctx.h.alpha(ev.logits), 1e-12)

    s_tr = np.sum(a_tr, axis=1)
    s_ev = np.sum(a_ev, axis=1)
    q_tr = a_tr / s_tr[:, None]
    q_ev = a_ev / s_ev[:, None]

    h_tr = -np.sum(q_tr * np.log(q_tr), axis=1)
    h_ev = -np.sum(q_ev * np.log(q_ev), axis=1)

    z_tr = np.log(s_tr)
    z_ev = np.log(s_ev)

    # Learn the in-scope relationship between total Dirichlet strength and
    # categorical-shape entropy; a positive residual is unusually diffuse.
    x_tr = np.column_stack((np.ones_like(z_tr), z_tr, z_tr * z_tr))
    ridge = 1e-5
    gram = x_tr.T @ x_tr + ridge * np.eye(x_tr.shape[1])
    beta = np.linalg.solve(gram, x_tr.T @ h_tr)

    expected_ev = np.column_stack(
        (np.ones_like(z_ev), z_ev, z_ev * z_ev)
    ) @ beta

    residual = h_ev - expected_ev
    return np.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)
