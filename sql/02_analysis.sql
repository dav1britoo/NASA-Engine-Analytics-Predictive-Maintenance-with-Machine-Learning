-- 1) Auditar volume de dados e quantidade de motores
SELECT split, COUNT(*) AS measurements, COUNT(DISTINCT engine_id) AS engines,
       MIN(cycle) AS min_cycle, MAX(cycle) AS max_cycle
FROM engine_readings WHERE dataset = 'FD001'
GROUP BY split ORDER BY split;

-- 2) Vida total simulada dos motores de treinamento (falha ocorre ao fim da trajetória)
SELECT engine_id, MAX(cycle) AS cycles_to_failure
FROM engine_readings
WHERE dataset = 'FD001' AND split = 'train'
GROUP BY engine_id ORDER BY cycles_to_failure ASC;

-- 3) Sensor 2 e sensor 11 nos últimos 30 ciclos, para visualização de tendências
WITH lifetime AS (
    SELECT engine_id, MAX(cycle) AS failure_cycle
    FROM engine_readings
    WHERE dataset='FD001' AND split='train'
    GROUP BY engine_id
)
SELECT r.engine_id, r.cycle, (l.failure_cycle - r.cycle) AS rul,
       r.sensor_2, r.sensor_11
FROM engine_readings r
JOIN lifetime l ON l.engine_id=r.engine_id
WHERE r.dataset='FD001' AND r.split='train'
  AND l.failure_cycle-r.cycle BETWEEN 0 AND 30
ORDER BY r.engine_id, r.cycle;

-- 4) Janela móvel: média de 10 leituras do sensor 11 por motor
SELECT engine_id, cycle, sensor_11,
       AVG(sensor_11) OVER (
          PARTITION BY engine_id ORDER BY cycle
          ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
       ) AS sensor_11_avg_10
FROM engine_readings
WHERE dataset='FD001' AND split='train'
ORDER BY engine_id, cycle;

-- 5) Risco relativo: rótulos para treino apenas, usando RUL conhecido.
-- IMPORTANTE: não é uma previsão de falha e não deve ser usado como classificação do teste.
WITH lifetime AS (
 SELECT engine_id, MAX(cycle) AS last_cycle
 FROM engine_readings WHERE dataset='FD001' AND split='train'
 GROUP BY engine_id
)
SELECT r.engine_id, r.cycle, (l.last_cycle-r.cycle) AS rul,
 CASE WHEN l.last_cycle-r.cycle <= 30 THEN 'critical'
      WHEN l.last_cycle-r.cycle <= 80 THEN 'warning'
      ELSE 'normal' END AS lifecycle_label
FROM engine_readings r JOIN lifetime l ON r.engine_id=l.engine_id
WHERE r.dataset='FD001' AND r.split='train'
ORDER BY r.engine_id, r.cycle;

-- 6) Para o teste: a RUL real fornecida se aplica ao último ciclo observado.
SELECT r.engine_id, MAX(r.cycle) AS final_observed_cycle, t.remaining_cycles
FROM engine_readings r JOIN engine_test_rul t
  ON t.dataset=r.dataset AND t.engine_id=r.engine_id
WHERE r.dataset='FD001' AND r.split='test'
GROUP BY r.engine_id, t.remaining_cycles
ORDER BY t.remaining_cycles;
