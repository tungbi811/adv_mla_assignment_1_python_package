import numpy as np
import itertools
import random
from typing import Dict, List, Any
from sklearn.base import clone
from sklearn.metrics import get_scorer
import joblib
from pathlib import Path


def grid_search(model, param_grid: Dict[str, List[Any]], X_train, y_train, X_val, y_val,scoring: str = "accuracy") -> List[Dict[str, Any]]:
    """
    Grid search over all param combinations with progress printing.
    """
    scorer = get_scorer(scoring)
    results = []

    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))
    total = len(combos)

    print(f"[GridSearch] {total} combinations to evaluate...")

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        print(f"  → ({i}/{total}) Testing params: {params}")

        m = clone(model).set_params(**params)
        m.fit(X_train, y_train)
        score = scorer(m, X_val, y_val)

        print(f"    Score ({scoring}): {score:.4f}")
        results.append({"params": params, "score": score})

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    print(f"[GridSearch] Best params: {results[0]['params']} | Best score: {results[0]['score']:.4f}")
    return results


def _sample_value(space_item, np_rng):
    # scipy.stats frozen distributions have .rvs
    if hasattr(space_item, "rvs"):
        # returns scalar; ensure Python type
        val = space_item.rvs(random_state=np_rng)
        # cast ints cleanly (randint can return np.int64)
        if hasattr(space_item, "dist") and space_item.dist.name == "randint":
            return int(val)
        return float(val) if isinstance(val, (np.floating,)) else val
    # callables (custom samplers)
    if callable(space_item):
        return space_item()
    # sequences / sets
    try:
        return np_rng.choice(list(space_item))
    except TypeError:
        # scalar already
        return space_item

def random_search(
    model,
    param_distributions: Dict[str, Any],
    X_train, y_train,
    X_val, y_val,
    n_iter: int = 10,
    scoring: str = "accuracy",
    seed: int = 42,
):
    """Random search that supports scipy.stats distributions and lists."""
    np_rng = np.random.default_rng(seed)
    scorer = get_scorer(scoring)
    results = []

    keys = list(param_distributions.keys())
    for i in range(1, n_iter + 1):
        params = {k: _sample_value(param_distributions[k], np_rng) for k in keys}
        print(f"  → ({i}/{n_iter}) Testing params: {params}")

        m = clone(model).set_params(**params)
        m.fit(X_train, y_train)
        score = scorer(m, X_val, y_val)
        print(f"    Score ({scoring}): {score:.4f}")
        results.append({"params": params, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    print(f"[RandomSearch] Best params: {results[0]['params']} | Best score: {results[0]['score']:.4f}")
    return results

def save_model(model, path: str):
    """Save a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model (e.g., sklearn, xgboost, etc.)
    path : str
        File path to save the model (e.g., "artifacts/model.pkl").
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[SaveModel] Model saved to {path}")


def load_model(path: str):
    """Load a trained model from disk.

    Parameters
    ----------
    path : str
        File path where the model is stored.

    Returns
    -------
    object
        The loaded model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    print(f"[LoadModel] Model loaded from {path}")
    return model