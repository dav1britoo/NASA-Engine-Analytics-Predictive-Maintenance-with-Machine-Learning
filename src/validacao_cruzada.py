
import os
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Conexão com o PostgreSQL
url = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/nasa_engines"
)

engine = create_engine(url)

df = pd.read_sql("""
    SELECT *
    FROM engine_readings
    WHERE dataset = 'FD001' AND split = 'train'
""", engine)

# Calcular RUL
df["vida_total"] = df.groupby("engine_id")["cycle"].transform("max")
df["RUL"] = (df["vida_total"] - df["cycle"]).clip(upper=125)

sensores = [f"sensor_{i}" for i in range(1, 22)]
settings = [f"setting_{i}" for i in range(1, 4)]

configuracoes = {
    "Somente ciclos": ["cycle"],
    "Somente sensores": settings + sensores,
    "Ciclos + sensores": ["cycle"] + settings + sensores
}

# Dividir por motor
kfold = GroupKFold(n_splits=5)
divisoes = list(kfold.split(df, groups=df["engine_id"]))

resultados = []

for nome, features in configuracoes.items():

    for rodada, (idx_treino, idx_val) in enumerate(divisoes, 1):

        treino = df.iloc[idx_treino]
        validacao = df.iloc[idx_val]

        modelo = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        modelo.fit(treino[features], treino["RUL"])

        previsoes = modelo.predict(validacao[features])

        mae = mean_absolute_error(validacao["RUL"], previsoes)
        rmse = mean_squared_error(
            validacao["RUL"], previsoes
        ) ** 0.5

        resultados.append({
            "Modelo": nome,
            "Rodada": rodada,
            "MAE": mae,
            "RMSE": rmse
        })

        print(
            f"{nome} | Rodada {rodada} | "
            f"MAE: {mae:.2f} | RMSE: {rmse:.2f}"
        )

# Resumo dos resultados
resultado_df = pd.DataFrame(resultados)

resumo = resultado_df.groupby("Modelo")[["MAE", "RMSE"]].agg(
    ["mean", "std"]
)

print("\nRESULTADOS DA VALIDAÇÃO CRUZADA")
print(resumo.round(2))

os.makedirs("outputs", exist_ok=True)

resultado_df.to_csv(
    "outputs/validacao_cruzada.csv",
    index=False
)

resumo.to_csv("outputs/resumo_validacao.csv")

# Gráfico de MAE por rodada
plt.figure(figsize=(11, 6))

for nome, grupo in resultado_df.groupby("Modelo"):
    plt.plot(
        grupo["Rodada"],
        grupo["MAE"],
        marker="o",
        linewidth=2,
        label=nome
    )

plt.title("NASA C-MAPSS | Validação cruzada por motor")
plt.xlabel("Rodada de validação")
plt.ylabel("MAE (ciclos)")
plt.xticks(range(1, 6))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/validacao_cruzada.png",
    dpi=300
)

plt.show()
