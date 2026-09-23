# fantasy-ml

Predicting weekly fantasy football points for the players in my ESPN league (full PPR), and checking whether a model can beat ESPN's own projections.

> **Status: work in progress.** Notebook 01 (data ingestion) is done; notebook 02 is in progress (K and D/ST scoring done). Modeling has not started yet, so there are no results to report.

## Stack

- **Python 3.11**, **Jupyter**
- [`nflreadpy`](https://github.com/nflverse/nflreadpy): weekly player stats from nflverse (2023–2026)
- [`espn-api`](https://github.com/cwendt94/espn-api): league settings, rosters and ESPN projections
- [`polars`](https://pola.rs/): data wrangling
- [`nbstripout`](https://github.com/kynan/nbstripout): keeps notebook outputs out of git

## Repository layout

```
config/scoring.yaml     # full league scoring rules (offense, K, D/ST)
config/validation.yaml  # walk-forward folds shared by the baseline and model notebooks
notebooks/01_datos.ipynb
notebooks/02_features.ipynb
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

## Validation

Walk-forward by season, training only on past seasons:

| fold | train | validate | compared against |
|---|---|---|---|
| 1 | 2023 | 2024 | rolling-average baseline |
| 2 | 2023–2024 | 2025 | rolling-average baseline |
| holdout | 2023–2025 | 2026 | ESPN projections |

The league was created in 2026, so ESPN projections exist only for 2026.

## Limitations

- **Closing betting lines.** nflverse schedules provide closing spreads and totals. Real lineup decisions are made earlier, with opening or current lines, so backtests are somewhat optimistic.
- **The model assumes the player plays.** Targets exist only for games played. The model predicts points *if active*; availability (injuries, inactives) is handled outside the model.
- **K and D/ST scoring** is computed from nflverse stats and validated against ESPN's actual 2026 points (K 64/64 games, D/ST 63/64; the one mismatch is a sack credited differently by the two sources). The same rules are assumed for 2023–2025.

## Roadmap

- [x] **01_datos**: download weekly stats, connect to the league, validate scoring rules, and join the roster with ESPN projections through nflverse IDs
- [ ] **02_features**
  - [x] K and D/ST points from `scoring.yaml`, validated against ESPN
  - [ ] rolling usage and matchup features without future data leaking into them
- [ ] **03_baseline**: rolling-average baseline on the walk-forward folds; ESPN projections on 2026
- [ ] **04_modelo**: per-position models evaluated on the same walk-forward folds, then compared against ESPN on 2026
- [ ] **05_lineup**: generate weekly predictions and start/sit recommendations for my roster
