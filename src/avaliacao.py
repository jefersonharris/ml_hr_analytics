# src/avaliacao.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    f1_score,
)
import shap
import numpy as np
import os


def plotar_matriz_confusao(y_true, y_pred, modelo_nome):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de Confusão - {modelo_nome}")
    os.makedirs("graficos", exist_ok=True)
    plt.savefig(f"graficos/matriz_confusao_{modelo_nome}.png")
    plt.close()


def plotar_curvas(y_true, y_scores, modelo_nome):
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision, label=f"{modelo_nome}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    os.makedirs("graficos", exist_ok=True)
    plt.savefig(f"graficos/pr_curve_{modelo_nome}.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, label=f"{modelo_nome}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"graficos/roc_curve_{modelo_nome}.png")
    plt.close()


def interpretabilidade_shap(modelo, X_sample):
    explainer = shap.Explainer(modelo)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False)
    os.makedirs("graficos", exist_ok=True)
    plt.savefig("graficos/shap_summary.png")
    plt.close()


def ajuste_threshold(y_true, y_scores):
    precisao, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precisao * recall) / (precisao + recall + 1e-6)
    melhor_idx = np.argmax(f1_scores)
    melhor_threshold = thresholds[melhor_idx]
    print(f"Melhor threshold com base no F1: {melhor_threshold:.2f}")
    return melhor_threshold
