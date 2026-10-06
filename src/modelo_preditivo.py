
import os
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump

# 1. Conectar ao PostgreSQL
url = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/nasa_engines"
)

engine = create_engine(url)

# 2. Carregar os dados
df = pd.read_sql("""
    SELECT *
    FROM engine_readings
    WHERE dataset = 'FD001'
""", engine)

rul_real = pd.read_sql("""
    SELECT engine_id, remaining_cycles
    FROM engine_test_rul
    WHERE dataset = 'FD001'
""", engine)

train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

# 3. Calcular a vida restante no treinamento
train["vida_total"] = (
    train.groupby("engine_id")["cycle"].transform("max")
)

train["RUL"] = train["vida_total"] - train["cycle"]

# Limitar a RUL de treinamento a 125 ciclos
# para priorizar a fase relevante de degradação
train["RUL_alvo"] = train["RUL"].clip(upper=125)

# 4. Selecionar variáveis
sensores = [f"sensor_{i}" for i in range(1, 22)]
settings = [f"setting_{i}" for i in range(1, 4)]

features = ["cycle"] + settings + sensores

X_train = train[features]
y_train = train["RUL_alvo"]

# Usar somente a última leitura disponível
# de cada motor do conjunto de teste
test_final = (
    test.sort_values("cycle")
    .groupby("engine_id")
    .tail(1)
    .sort_values("engine_id")
    .copy()
)

X_test = test_final[features]

# 5. Treinar o modelo
modelo = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

print("Treinando Random Forest...")
modelo.fit(X_train, y_train)

# 6. Fazer previsões
previsoes = modelo.predict(X_test)

resultados = test_final[["engine_id"]].copy()
resultados["RUL_prevista"] = previsoes

resultados = resultados.merge(
    rul_real,
    on="engine_id",
    how="inner",
    validate="one_to_one"
)

if len(resultados) != len(test_final):
    raise ValueError("Faltam valores reais de RUL para alguns motores.")

y_real = resultados["remaining_cycles"]
y_pred = resultados["RUL_prevista"]

# 7. Avaliar
mae = mean_absolute_error(y_real, y_pred)
rmse = mean_squared_error(y_real, y_pred) ** 0.5

print("\nRESULTADOS DO MODELO")
print(f"MAE: {mae:.2f} ciclos")
print(f"RMSE: {rmse:.2f} ciclos")

# Comparar com uma previsão constante
baseline = [y_train.median()] * len(y_real)
mae_baseline = mean_absolute_error(y_real, baseline)

print(f"MAE baseline: {mae_baseline:.2f} ciclos")

# 8. Salvar resultados
os.makedirs("outputs", exist_ok=True)

resultados.to_csv(
    "outputs/previsoes_rul.csv",
    index=False
)

dump(modelo, "outputs/modelo_random_forest.joblib")

# 9. Gráfico de previsões
plt.figure(figsize=(9, 7))

plt.scatter(
    y_real,
    y_pred,
    alpha=0.65,
    s=35
)

limite = max(y_real.max(), y_pred.max())

plt.plot(
    [0, limite],
    [0, limite],
    "--",
    label="Previsão perfeita"
)

plt.xlabel("RUL real (ciclos)")
plt.ylabel("RUL prevista (ciclos)")
plt.title("NASA C-MAPSS | Random Forest: RUL real vs prevista")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "outputs/previsao_rul.png",
    dpi=300
)

plt.show()

print("\nArquivos salvos na pasta outputs.")
