
from joblib import Parallel, delayed
import numpy as np

def kfold_indices(n_samples:int, k:int, seed:int=42):
    rng = np.random.default_rng(seed); idx = np.arange(n_samples); rng.shuffle(idx)
    return np.array_split(idx, k)

def parallel_cv_fit_score(build_model, X, y, params_list, k=5, n_jobs=-1, scoring=None, seed=42):
    folds = kfold_indices(len(X), k, seed=seed)
    def _one_cfg(params):
        scores = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j!=i])
            model = build_model(params)
            model.fit(X[train_idx], y[train_idx])
            s = model.score(X[val_idx], y[val_idx]) if scoring is None else scoring(model, X[val_idx], y[val_idx])
            scores.append(s)
        return params, float(np.mean(scores))
    results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one_cfg)(p) for p in params_list)
    results.sort(key=lambda t: t[1], reverse=True)
    return results
