from data.sets import load_data
from models.training import grid_search, random_search
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)
X_train, y_train, X_val, y_val, X_test, y_test = load_data(r"processed")

param_space = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 15, 20],
}

best_rand = random_search(model, param_space, X_train, y_train, X_val, y_val, n_iter=5, scoring="roc_auc")
print(best_rand[0])