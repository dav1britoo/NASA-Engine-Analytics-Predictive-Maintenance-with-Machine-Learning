
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

url = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@127.0.0.1:5432/nasa_engines"
)

engine = create_engine(url)

query = """
SELECT engine_id, cycle,
       sensor_2, sensor_4, sensor_7, sensor_11
FROM engine_readings
WHERE split = 'train'
"""

df = pd.read_sql(query, engine)

df["vida_total"] = df.groupby("engine_id")["cycle"].transform("max")
df["vida_percentual"] = 100 * df["cycle"] / df["vida_total"]
df["faixa"] = (df["vida_percentual"] / 5).astype(int).clip(upper=19)

sensores = ["sensor_2", "sensor_4", "sensor_7", "sensor_11"]

# Média de cada sensor por motor e faixa
por_motor = df.groupby(["engine_id", "faixa"])[sensores].mean()
media = por_motor.groupby("faixa").mean()

fig, axes = plt.subplots(2, 2, figsize=(13, 8))

for ax, sensor in zip(axes.flatten(), sensores):
    ax.plot(media.index * 5 + 2.5, media[sensor], linewidth=2)
    ax.set_title(sensor.replace("_", " ").title())
    ax.set_xlabel("Vida operacional consumida (%)")
    ax.set_ylabel("Leitura média")
    ax.grid(alpha=0.3)

fig.suptitle(
    "NASA C-MAPSS | Degradação média de 100 motores",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/comparacao_sensores_100_motores.png", dpi=300)
plt.show()
