"""Load NASA C-MAPSS FD001 text files into PostgreSQL.

Usage: pip install -r requirements.txt
       export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/nasa_engines
       python src/load_data.py
Dataset files must be extracted into data/ (or data/CMAPSSData/).
"""
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIELDS = ['engine_id', 'cycle'] + [f'setting_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]


def find_file(filename):
    files = list(DATA.rglob(filename))
    if not files:
        raise FileNotFoundError(f'Missing {filename}. Extract the NASA CMAPSSData.zip into {DATA}')
    return files[0]


def main():
    url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/nasa_engines')
    engine = create_engine(url)
    schema = (ROOT / 'sql' / '01_schema.sql').read_text(encoding='utf-8')
    with engine.begin() as conn:
        for statement in schema.split(';'):
            if statement.strip():
                conn.execute(text(statement))
        conn.execute(text("DELETE FROM engine_readings WHERE dataset='FD001'"))
        conn.execute(text("DELETE FROM engine_test_rul WHERE dataset='FD001'"))

    for split in ('train', 'test'):
        path = find_file(f'{split}_FD001.txt')
        frame = pd.read_csv(path, sep=r'\s+', header=None)
        if frame.shape[1] != 26:
            raise ValueError(f'{path}: expected 26 columns, got {frame.shape[1]}')
        frame.columns = FIELDS
        frame.insert(0, 'dataset', 'FD001')
        frame.insert(1, 'split', split)
        frame.to_sql('engine_readings', engine, if_exists='append', index=False, chunksize=3000, method='multi')
        print(f'Imported {split}: {len(frame)} sensor readings, {frame.engine_id.nunique()} engines')

    rul_path = find_file('RUL_FD001.txt')
    rul = pd.read_csv(rul_path, sep=r'\s+', header=None)
    if rul.shape[1] != 1:
        raise ValueError('RUL_FD001.txt must have one column')
    rul.columns = ['remaining_cycles']
    rul.insert(0, 'dataset', 'FD001')
    rul.insert(1, 'engine_id', range(1, len(rul) + 1))
    rul.to_sql('engine_test_rul', engine, if_exists='append', index=False)
    print(f'Imported test RUL targets for {len(rul)} engines')

if __name__ == '__main__':
    main()
