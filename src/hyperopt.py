# src/hyperopt.py
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def _objective(trial, X, y):
    # Hiperparâmetros distribuídos
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 16, 255),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
    }

    model = LGBMClassifier(**params)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return f1_score(y_valid, preds)


def otimizar_lightgbm(X, y, n_trials: int = 50):
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: _objective(trial, X, y), n_trials=n_trials)
    print("🏆  Melhor F1:", study.best_value)
    print("🔧  Parâmetros ótimos:", study.best_params)
    # Treina modelo final com os melhores parâmetros
    best_model = LGBMClassifier(**study.best_params, random_state=42)
    best_model.fit(X, y)
    return best_model
