"""Shared frequency-weighted Ridge implementation; no model inference."""
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import numpy as np
import pandas as pd

def ridge_r2_batch(df, numeric, categorical, weights, variance=None):
    """Frequency-weighted Ridge CV, alpha=1, fixed folds and target variance.

    Work in raw numeric coordinates. The penalty on a raw numeric coefficient
    is its training-fold weighted variance; this is algebraically equivalent
    to StandardScaler followed by Ridge with an unpenalized intercept.
    Full indicator columns have penalty one. No missing values accepted.
    """
    weights = np.atleast_2d(weights)
    num = df[numeric].to_numpy(float)
    cat = pd.get_dummies(df[categorical].astype(str), dtype=float).to_numpy()
    x = np.column_stack((num, cat))
    y = df.CIE.to_numpy(float)
    assert np.isfinite(x).all() and np.isfinite(y).all()
    fold = df.fold_id.to_numpy()
    variance = np.var(y) if variance is None else variance
    sse = np.zeros(len(weights))
    k = x.shape[1]
    diag = np.arange(k)
    for f in range(5):
        tr = fold != f
        te = ~tr
        w = weights[:, tr]
        sw = w.sum(axis=1)
        assert (sw > 0).all() and (weights[:, te].sum(axis=1) > 0).all()
        xt, yt = x[tr], y[tr]
        xm = (w @ xt) / sw[:, None]
        ym = (w @ yt) / sw
        outer = (xt[:, :, None] * xt[:, None, :]).reshape(len(xt), -1)
        gram = (w @ outer).reshape(-1, k, k) - sw[:, None, None] * xm[:, :, None] * xm[:, None, :]
        rhs = w @ (xt * yt[:, None]) - sw[:, None] * xm * ym[:, None]
        nv = (w @ (xt[:, :len(numeric)] ** 2)) / sw[:, None] - xm[:, :len(numeric)] ** 2
        nv = np.maximum(nv, 0)
        penalty = np.ones((len(w), k))
        penalty[:, :len(numeric)] = np.where(nv < 1e-24, 1, nv)
        gram[:, diag, diag] += penalty
        beta = np.linalg.solve(gram, rhs[:, :, None])[:, :, 0]
        pred = beta @ x[te].T + (ym - np.sum(beta * xm, axis=1))[:, None]
        sse += np.sum(weights[:, te] * (y[te] - pred) ** 2, axis=1)
    return 1 - sse / (weights.sum(axis=1) * variance)

