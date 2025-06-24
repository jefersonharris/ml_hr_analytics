# Projeto de Machine Learning - HR Analytics Challenge

> MBA Engenharia de Dados - Disciplina: Data Science Experience
> Professor: Matheus H. P. Pacheco
> Entrega: 17/07/2025

---

## ✨ 1. Resumo Executivo

A TechCorp Brasil enfrenta um aumento de 35% na taxa de rotatividade de colaboradores, resultando em um impacto financeiro de R\$ 45 milhões. O presente projeto implementa um sistema preditivo de Machine Learning para identificar colaboradores com risco elevado de pedir demissão (attrition), permitindo que o RH tome ações preventivas.

O modelo Random Forest atingiu os melhores resultados com métricas robustas (F1-score e Precision-Recall AUC).

---

## 🔍 2. Introdução

* **Contexto:** A alta rotatividade de funcionários afeta produtividade, moral e gera custos elevados.
* **Objetivo:** Desenvolver um pipeline preditivo de ML para prever `Attrition` com foco em aplicação real.
* **Metodologia:** Análise exploratória ➞ Feature Engineering ➞ Treinamento de modelos ➞ Interpretação SHAP ➞ Recomendacões de negócio.

---

## 📊 3. Análise Exploratória

* Dataset com **1470 registros** e **35 variáveis**.
* Target `Attrition` com distribuição desbalanceada (\~16% positivos).
* Principais insights:

  * Funcionários que trabalham em regime de OverTime e moram longe pedem mais demissão.
  * Baixa satisfação e baixo envolvimento estão correlacionados ao `Attrition`.

### 📅 Gráficos:

* `graficos/distribuicao_attrition.png`
* `graficos/mapa_correlacao.png`

---

## 🪨 4. Desenvolvimento da Solução

### 🔹 Feature Engineering

Criadas 10+ variáveis derivadas com lógica de negócio:

* `EngagementScore`, `AvgYearsPerRole`, `IncomePerYear`, `TravelPenalty`, `HasStockOption`, entre outras.

### 📊 Modelos Utilizados

* `LogisticRegression`
* `RandomForestClassifier`
* `XGBClassifier`
* `CatBoostClassifier`

### ⚖️ Balanceamento:

* Aplicado `stratify` na divisão treino/teste.
* Em versão anterior, usado `SMOTE` para balanceamento completo.

---

## ⚖️ 5. Resultados e Avaliação

### 🏋️ Métricas:

* **Random Forest** teve o melhor F1 e curva PR.
* Precision-Recall AUC e threshold ideal calculado.

### 🔄 Interpretação:

* Gráfico SHAP com as variáveis que mais impactam a predição de `Attrition`.

### 📅 Gráficos:

* `graficos/matriz_confusao_*.png`
* `graficos/pr_curve_*.png`
* `graficos/roc_curve_*.png`
* `graficos/shap_summary.png`

---

## 🚀 6. Implementação e Próximos Passos

* Estrutura modular com scripts `main.py`, `avaliar_modelos.py`, `src/`
* Avaliação isolada de modelos com `joblib`
* Potencial para deploy com Streamlit ou API (bônus)
* Recomendação: monitoramento contínuo com alertas no RH

---

## ✅ 7. Conclusão

* Projeto entrega valor real ao RH da TechCorp
* Redução esperada de até 10% no `Attrition` com ações preventivas
* Próximos passos incluem deploy e integração com sistemas internos

---

> "In God we trust. All others must bring data." – W. Edwards Deming
