CREATE TABLE IF NOT EXISTS engine_readings (
    dataset TEXT NOT NULL DEFAULT 'FD001',
    split TEXT NOT NULL CHECK (split IN ('train', 'test')),
    engine_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    setting_1 DOUBLE PRECISION,
    setting_2 DOUBLE PRECISION,
    setting_3 DOUBLE PRECISION,
    sensor_1 DOUBLE PRECISION,
    sensor_2 DOUBLE PRECISION,
    sensor_3 DOUBLE PRECISION,
    sensor_4 DOUBLE PRECISION,
    sensor_5 DOUBLE PRECISION,
    sensor_6 DOUBLE PRECISION,
    sensor_7 DOUBLE PRECISION,
    sensor_8 DOUBLE PRECISION,
    sensor_9 DOUBLE PRECISION,
    sensor_10 DOUBLE PRECISION,
    sensor_11 DOUBLE PRECISION,
    sensor_12 DOUBLE PRECISION,
    sensor_13 DOUBLE PRECISION,
    sensor_14 DOUBLE PRECISION,
    sensor_15 DOUBLE PRECISION,
    sensor_16 DOUBLE PRECISION,
    sensor_17 DOUBLE PRECISION,
    sensor_18 DOUBLE PRECISION,
    sensor_19 DOUBLE PRECISION,
    sensor_20 DOUBLE PRECISION,
    sensor_21 DOUBLE PRECISION,
    PRIMARY KEY (dataset, split, engine_id, cycle)
);

CREATE TABLE IF NOT EXISTS engine_test_rul (
    dataset TEXT NOT NULL DEFAULT 'FD001',
    engine_id INTEGER NOT NULL,
    remaining_cycles INTEGER NOT NULL CHECK (remaining_cycles >= 0),
    PRIMARY KEY (dataset, engine_id)
);

CREATE INDEX IF NOT EXISTS idx_readings_split_cycle ON engine_readings(dataset, split, cycle);
