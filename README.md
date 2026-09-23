# fantasy-ml

Predicting weekly fantasy football points for the players in my ESPN league (full PPR), and checking whether a model can beat ESPN's own projections.

> **Status: work in progress.** Notebook 01 (data ingestion) is done. Modeling has not started yet, so there are no results to report.

## Stack

- **Python 3.11**, **Jupyter**
- [`nflreadpy`](https://github.com/nflverse/nflreadpy): weekly player stats from nflverse (2023–2026)
- [`espn-api`](https://github.com/cwendt94/espn-api): league settings, rosters and ESPN projections
- [`polars`](https://pola.rs/): data wrangling
- [`nbstripout`](https://github.com/kynan/nbstripout): keeps notebook outputs out of git

## Repository layout

```
config/scoring.yaml     # full league scoring rules (offense, K, D/ST)
notebooks/01_datos.ipynb
.env.example            # ESPN credentials template (the real .env is gitignored)
requirements.txt
```

## Reproducing

```bash
git clone https://github.com/romogil99-max/fantasy-ml.git
cd fantasy-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Strip notebook outputs automatically on every commit (run once per clone)
nbstripout --install

# ESPN credentials (needed for a private league)
cp .env.example .env    # then fill in ESPN_LEAGUE_ID, ESPN_S2, ESPN_SWID, ESPN_TEAM_ID
```

`.env.example` explains where to find each value. Run the notebooks from the `notebooks/` directory. Downloaded data is cached in `data/`, which is gitignored.

Notebook 01 checks that `config/scoring.yaml` matches the league's live settings and stops if it doesn't. To use a different league, update that file first.

## Roadmap

- [x] **01_datos**: download weekly stats, connect to the league, validate scoring rules, and join the roster with ESPN projections through nflverse IDs
- [ ] **02_features**: compute K and D/ST points from `scoring.yaml`, then build rolling usage and matchup features without future data leaking into them
- [ ] **03_baseline**: evaluate ESPN projections and simple baselines (rolling averages)
- [ ] **04_modelo**: train per-position models with time-based validation and compare them against the ESPN baseline
- [ ] **05_lineup**: generate weekly predictions and start/sit recommendations for my roster
