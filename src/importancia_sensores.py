
import os
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from joblib import load
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error

# Conectar ao PostgreSQL
url = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/nasa_engines"
)

engine = create_engine(url)

# Recuperar modelo treinado
modelo = load("outputs/modelo_random_forest.joblib")

# Carregar motores de teste e RUL real
df = pd.read_sql("""
    SELECT *
    FROM engine_readings
    WHERE dataset = 'FD001'
      AND split = 'test'
""", engine)

rul = pd.read_sql("""
    SELECT engine_id, remaining_cycles
    FROM engine_test_rul
    WHERE dataset = 'FD001'
""", engine)

# Última leitura de cada motor
test_final = (
    df.sort_values("cycle")
    .groupby("engine_id")
    .tail(1)
    .sort_values("engine_id")
    .copy()
)

test_final = test_final.merge(
    rul,
    on="engine_id",
    validate="one_to_one"
)

sensores = [f"sensor_{i}" for i in range(1, 22)]
settings = [f"setting_{i}" for i in range(1, 4)]
features = ["cycle"] + settings + sensores

X = test_final[features]
y = test_final["remaining_cycles"]

# Calcular importância por permutação
resultado = permutation_importance(
    modelo,
    X,
    y,
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importancia = pd.DataFrame({
    "variavel": features,
    "importancia": resultado.importances_mean,
    "desvio": resultado.importances_std
})

importancia = importancia.sort_values(
    "importancia",
    ascending=False
)

print("\nTOP 10 VARIÁVEIS MAIS IMPORTANTES")
print(importancia.head(10).to_string(index=False))

# Criar gráfico
top10 = importancia.head(10).sort_values("importancia")

plt.figure(figsize=(11, 6))
plt.barh(
    top10["variavel"],
    top10["importancia"]
)

plt.xlabel("Aumento médio do MAE ao embaralhar a variável (ciclos)")
plt.title("NASA C-MAPSS | Importância das variáveis — Random Forest")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

os.makedirs("outputs", exist_ok=True)

plt.savefig(
    "outputs/importancia_sensores.png",
    dpi=300
)

importancia.to_csv(
    "outputs/importancia_variaveis.csv",
    index=False
)

plt.show()
