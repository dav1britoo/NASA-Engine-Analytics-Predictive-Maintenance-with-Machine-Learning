# NASA Engine Health Analytics

SQL-based analysis of synthetic aircraft engine degradation using the **NASA C-MAPSS FD001** dataset.

## Data source

- NASA Open Data: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- Direct ZIP: https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- Original authors: A. Saxena and K. Goebel (2008), NASA Ames Prognostics Data Repository.

The dataset is *simulated*, not actual flight logs. FD001 has 100 training engines and 100 test engines; each reading contains engine ID, cycle, 3 settings, and 21 sensor measurements. Training trajectories extend to failure. Testing trajectories are truncated, and `RUL_FD001.txt` gives true remaining useful life at the **last observed test cycle**.

## Run on Windows (PowerShell) or terminal

1. Install Docker Desktop and Python 3.10+.
2. Download the NASA ZIP at the link above and extract into `data/`. The loader searches recursively, so `data/CMAPSSData/train_FD001.txt` is fine.
3. Start PostgreSQL:
   ```bash
   docker compose up -d
   ```
4. Install dependencies and load the data:
   ```bash
   pip install -r requirements.txt
   python src/load_data.py
   ```
5. Use DBeaver or pgAdmin to connect to `localhost:5432`, database `nasa_engines`, user/password `postgres`/`postgres`, and execute SQL in `sql/02_analysis.sql`.

## Structure

- `sql/01_schema.sql`: PostgreSQL schema and index.
- `sql/02_analysis.sql`: six exploratory queries with CTEs and window functions.
- `src/load_data.py`: imports training, testing and ground-truth RUL.
- `data/`: your locally extracted dataset (excluded from version control).

## Questions to answer

1. How long does each training engine operate before simulated failure?
2. Which sensors show trends as RUL approaches zero?
3. Can moving averages separate short-term noise from gradual degradation?
4. How does degradation differ across engines?
5. How well can a **future** predictive model estimate RUL on held-out test engines?

## Important methodology

Labels calculated from `MAX(cycle)` apply **only to training run-to-failure data**. Do not infer a test engine's true life from its final observed cycle. For evaluation, compare a model prediction at each test engine's final observed cycle to `engine_test_rul.remaining_cycles`. Avoid including the future failure cycle as a predictive input, and split by engine rather than randomly by row to prevent leakage.

## Next milestones

- Phase 1: SQL data audit + exploratory analysis.
- Phase 2: Dashboards (Power BI or Streamlit).
- Phase 3: Baseline RUL regression with engine-level validation; report MAE and RMSE against test ground truth.
- Phase 4: Document evidence-based findings and publish to GitHub and LinkedIn.
