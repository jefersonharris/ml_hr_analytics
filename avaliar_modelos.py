# avaliar_modelos.py - Avaliação isolada de modelos treinados

from src.avaliacao import (
    plotar_matriz_confusao,
    plotar_curvas,
    interpretabilidade_shap,
    ajuste_threshold,
)
from src.features import gerar_features
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Carregamento do dataset e features
    caminho_csv = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(caminho_csv)
    df_features = gerar_features(df).dropna()

    X = df_features.drop("Attrition", axis=1)
    y = df_features["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Carregamento de modelo salvo (ajuste conforme necessidade)
    modelo = joblib.load("modelos/modelo_rf.pkl")

    y_pred = modelo.predict(X_test)
    y_scores = modelo.predict_proba(X_test)[:, 1]

    nome_modelo = "RandomForest"
    plotar_matriz_confusao(y_test, y_pred, nome_modelo)
    plotar_curvas(y_test, y_scores, nome_modelo)
    interpretabilidade_shap(
        modelo, X_test.sample(min(100, len(X_test)), random_state=42)
    )
    ajuste_threshold(y_test, y_scores)

    print("\nAvaliação do modelo finalizada com sucesso.")
