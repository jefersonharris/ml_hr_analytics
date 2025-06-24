import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def executar_eda(caminho_csv):
    df = pd.read_csv(caminho_csv)

    print("\n=== Dimensões do Dataset ===")
    print(df.shape)

    print("\n=== Amostra dos Dados ===")
    print(df.head())

    print("\n=== Informações gerais ===")
    print(df.head())

    print("\n=== Distribuição da variavel Attrition ===")
    print(df["Attrition"].value_counts(normalize=True))

    # Criar pasta para salvar gráficos se ela ainda não existir
    os.makedirs("graficos", exist_ok=True)

    # Gráfico de distribuição da variável Target
    sns.countplot(x="Attrition", data=df)
    plt.title("Distribuição da Variável Target (Attrition)")
    plt.savefig("graficos/distribuicao_attrition.png")
    plt.close()

    # Estatísticas descritivas
    print("\n=== Estatísticas Descritivas ===")
    print(df.describe())

    # Mapa de correlação
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Mapa de Correlação entre Variáveis Numéricas")
    plt.savefig("graficos/mapa_correlacao.png")
    plt.close()

    print("\nEDA finalizada. Gráficos salvos na pasta 'graficos'.")
