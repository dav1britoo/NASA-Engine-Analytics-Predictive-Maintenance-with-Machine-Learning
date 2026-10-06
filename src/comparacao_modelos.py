
import os
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

# Conectar ao PostgreSQL
url = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/nasa_engines"
)
engine = create_engine(url)

# Carregar dados de treinamento
df = pd.read_sql("""
    SELECT *
    FROM engine_readings
    WHERE dataset = 'FD001' AND split = 'train'
""", engine)

# Calcular vida útil restante
df["vida_total"] = df.groupby("engine_id")["cycle"].transform("max")
df["RUL"] = (df["vida_total"] - df["cycle"]).clip(upper=125)

# Separar motores diferentes para treinamento e validação
divisor = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

idx_train, idx_val = next(
    divisor.split(df, groups=df["engine_id"])
)

treino = df.iloc[idx_train]
validacao = df.iloc[idx_val]

sensores = [f"sensor_{i}" for i in range(1, 22)]
settings = [f"setting_{i}" for i in range(1, 4)]

configuracoes = {
    "Somente ciclos": ["cycle"],
    "Somente sensores": settings + sensores,
    "Ciclos + sensores": ["cycle"] + settings + sensores
}

resultados = []

for nome, variaveis in configuracoes.items():

    modelo = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    modelo.fit(
        treino[variaveis],
        treino["RUL"]
    )

    previsoes = modelo.predict(validacao[variaveis])

    mae = mean_absolute_error(validacao["RUL"], previsoes)
    rmse = mean_squared_error(validacao["RUL"], previsoes) ** 0.5

    resultados.append({
        "Modelo": nome,
        "MAE": mae,
        "RMSE": rmse
    })

    print(f"{nome}: MAE={mae:.2f} | RMSE={rmse:.2f}")

# Organizar resultados
resultado_df = pd.DataFrame(resultados)

os.makedirs("outputs", exist_ok=True)

resultado_df.to_csv(
    "outputs/comparacao_modelos.csv",
    index=False
)

# Criar gráfico comparativo
ax = resultado_df.plot(
    x="Modelo",
    y=["MAE", "RMSE"],
    kind="bar",
    figsize=(11, 6),
    rot=0
)

ax.set_title("NASA C-MAPSS | Comparação de modelos Random Forest")
ax.set_ylabel("Erro (ciclos)")
ax.set_xlabel("")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/comparacao_modelos.png", dpi=300)
plt.show()
