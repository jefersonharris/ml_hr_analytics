from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd


def treinar_modelos(df):
    df = df.dropna()
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # Balanceamento com SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Separar treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )
    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
        "CatBoost": CatBoostClassifier(verbose=0),
    }

    resultados = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        print(f"\n=== Resultados do Modelo: {nome} ===")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        score = cross_val_score(modelo, X_res, y_res, cv=5, scoring="f1")
        resultados[nome] = score.mean()

    return resultados
