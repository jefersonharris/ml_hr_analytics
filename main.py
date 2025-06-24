# main.py - Pipeline central do projeto HR Analytics

from src.eda import executar_eda
from src.features import gerar_features
from src.avaliacao import (
    plotar_matriz_confusao,
    plotar_curvas,
    interpretabilidade_shap,
    ajuste_threshold,
)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import os

if __name__ == "__main__":
    caminho_csv = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

    # Etapa 1: Análise Exploratória de Dados (EDA)
    executar_eda(caminho_csv)

    # Etapa 2: Feature Engineering
    df = pd.read_csv(caminho_csv)
    df_features = gerar_features(df).dropna()

    # Etapa 3: Separação dos dados (com estratificação)
    X = df_features.drop("Attrition", axis=1)
    y = df_features["Attrition"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Etapa 4: Avaliação de múltiplos modelos
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    }

    os.makedirs("modelos", exist_ok=True)

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_scores = modelo.predict_proba(X_test)[:, 1]

        print(f"\nAvaliação - {nome}")
        plotar_matriz_confusao(y_test, y_pred, nome)
        plotar_curvas(y_test, y_scores, nome)

        if nome == "RandomForest":
            interpretabilidade_shap(
                modelo, X_test.sample(min(100, len(X_test)), random_state=42)
            )
            joblib.dump(modelo, "modelos/modelo_rf.pkl")

        ajuste_threshold(y_test, y_scores)

    print("\nPipeline finalizado com sucesso para todos os modelos.")
