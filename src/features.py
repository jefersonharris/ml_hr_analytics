import pandas as pd
from sklearn.preprocessing import LabelEncoder


def gerar_features(df):
    df_result = df.copy()

    # Convertendo a coluna 'Attrition' para binário
    df_result["Attrition"] = df_result["Attrition"].map({"Yes": 1, "No": 0})

    # Tempo médio por cargo
    df_result["AvgYearsPerRole"] = df_result["YearsAtCompany"] / (
        df_result["NumCompaniesWorked"] + 1
    )

    # Score de engajamento
    df_result["EngagementScore"] = df_result[
        ["JobInvolvement", "JobSatisfaction", "WorkLifeBalance"]
    ].mean(axis=1)

    # Renda proporcional ao tempo na empresa
    df_result["IncomePerYear"] = df_result["MonthlyIncome"] / (
        df_result["YearsAtCompany"] + 1
    )

    # Distância penalizada por frequência de viagens
    travel_map = {"Non-Travel": 0, "travel_Rarely": 1, "travel_Frequently": 2}
    df_result["BusinessTravelScore"] = df_result["BusinessTravel"].map(travel_map)
    df_result["TravelPenalty"] = (
        df_result["BusinessTravelScore"] * df_result["DistanceFromHome"]
    )

    # Tem participação acionária?
    df_result["HasStockOption"] = df_result["StockOptionLevel"].apply(
        lambda x: 1 if x > 0 else 0
    )

    # Senioridade realtiva (nível x anos de experiência)
    df_result["SeniorityIndex"] = df_result["JobLevel"] * df_result["YearsAtCompany"]

    # Participação em treinamentos por ano
    df_result["TrainingPerYear"] = df_result["TrainingTimesLastYear"] / (
        df_result["YearsAtCompany"] + 1
    )

    # Codificação simples de variáveis categóricas
    cat_cols = df_result.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        le = LabelEncoder()
        df_result[col] = le.fit_transform(df_result[col])

    return df_result
