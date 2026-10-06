
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Conexão com o PostgreSQL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/nasa_engines"
)

engine = create_engine(DATABASE_URL)

# Buscar dados do motor 1
query = """
SELECT cycle, sensor_2, sensor_4, sensor_7, sensor_11
FROM engine_readings
WHERE split = 'train' AND engine_id = 1
ORDER BY cycle
"""

df = pd.read_sql(query, engine)

# Criar gráficos
sensores = ["sensor_2", "sensor_4", "sensor_7", "sensor_11"]

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for ax, sensor in zip(axes.flatten(), sensores):
    ax.plot(df["cycle"], df[sensor], linewidth=1.5)
    ax.set_title(sensor.replace("_", " ").title())
    ax.set_xlabel("Ciclo de operação")
    ax.set_ylabel("Leitura do sensor")
    ax.grid(alpha=0.3)

fig.suptitle(
    "NASA C-MAPSS | Evolução dos sensores — Motor 1",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

# Salvar imagem para o portfólio
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/evolucao_sensores.png", dpi=300)
plt.show()

print("Gráfico salvo em outputs/evolucao_sensores.png")
