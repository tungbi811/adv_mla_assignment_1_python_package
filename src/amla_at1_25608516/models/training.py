# import numpy as np
# import itertools
# import random
# from typing import Dict, List, Any
# from sklearn.base import clone
# from sklearn.metrics import get_scorer
# import joblib
# from pathlib import Path


# def grid_search(model, param_grid: Dict[str, List[Any]], X_train, y_train, X_val, y_val,scoring: str = "accuracy") -> List[Dict[str, Any]]:
#     """
#     Grid search over all param combinations with progress printing.
#     """
#     scorer = get_scorer(scoring)
#     results = []

#     keys, values = zip(*param_grid.items())
#     combos = list(itertools.product(*values))
#     total = len(combos)

#     print(f"[GridSearch] {total} combinations to evaluate...")

#     for i, combo in enumerate(combos, 1):
#         params = dict(zip(keys, combo))
#         print(f"  → ({i}/{total}) Testing params: {params}")

#         m = clone(model).set_params(**params)
#         m.fit(X_train, y_train)
#         score = scorer(m, X_val, y_val)

#         print(f"    Score ({scoring}): {score:.4f}")
#         results.append({"params": params, "score": score})

#     results = sorted(results, key=lambda x: x["score"], reverse=True)
#     print(f"[GridSearch] Best params: {results[0]['params']} | Best score: {results[0]['score']:.4f}")
#     return results


# def random_search(
#     model,
#     param_distributions: Dict[str, List[Any]],
#     X_train, y_train,
#     X_val, y_val,
#     n_iter: int = 10,
#     scoring: str = "accuracy",
#     seed: int = 42,
# ) -> List[Dict[str, Any]]:
#     """Random search over param space with progress printing."""
#     rng = random.Random(seed)
#     scorer = get_scorer(scoring)
#     results = []

#     keys = list(param_distributions.keys())
#     print(f"[RandomSearch] {n_iter} random combinations to evaluate...")

#     for i in range(1, n_iter + 1):
#         params = {k: rng.choice(param_distributions[k]) for k in keys}
#         print(f"  → ({i}/{n_iter}) Testing params: {params}")

#         m = clone(model).set_params(**params)
#         m.fit(X_train, y_train)
#         score = scorer(m, X_val, y_val)

#         print(f"    Score ({scoring}): {score:.4f}")
#         results.append({"params": params, "score": score})

#     results = sorted(results, key=lambda x: x["score"], reverse=True)
#     print(f"[RandomSearch] Best params: {results[0]['params']} | Best score: {results[0]['score']:.4f}")
#     return results

# def save_model(model, path: str):
#     """Save a trained model to disk.

#     Parameters
#     ----------
#     model : object
#         Trained model (e.g., sklearn, xgboost, etc.)
#     path : str
#         File path to save the model (e.g., "artifacts/model.pkl").
#     """
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, path)
#     print(f"[SaveModel] Model saved to {path}")


# def load_model(path: str):
#     """Load a trained model from disk.

#     Parameters
#     ----------
#     path : str
#         File path where the model is stored.

#     Returns
#     -------
#     object
#         The loaded model.
#     """
#     path = Path(path)
#     if not path.exists():
#         raise FileNotFoundError(f"Model file not found: {path}")
#     model = joblib.load(path)
#     print(f"[LoadModel] Model loaded from {path}")
#     return model