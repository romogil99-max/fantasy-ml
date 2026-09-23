# fantasy-ml

Predicting weekly fantasy football points for the players in my ESPN league (full PPR), and checking whether a model can beat ESPN's own projections.

> **Status: work in progress.** Data, features, a weekly walk-forward backtest and a weekly predictions log are done. The 2026 comparison against ESPN is accumulating week by week in `data/predictions_log/`.

## Stack

- **Python 3.11**, **Jupyter**
- [`nflreadpy`](https://github.com/nflverse/nflreadpy): weekly player stats, snaps, expected points, schedules and rosters from nflverse (2023–2026)
- [`espn-api`](https://github.com/cwendt94/espn-api): league settings, rosters and ESPN projections
- [`polars`](https://pola.rs/): data wrangling
- [`LightGBM`](https://lightgbm.readthedocs.io/): gradient-boosted trees
- [`nbstripout`](https://github.com/kynan/nbstripout): keeps notebook outputs out of git

## Repository layout

```
src/fantasy_ml/          # shared code used by every notebook
  data.py                #   paths, config and cached nflverse downloads
  scoring.py             #   K and D/ST fantasy points (validated against ESPN)
  features.py            #   leak-free features, including rows for the upcoming week
  model.py               #   LightGBM, baseline, weekly walk-forward, hyperparameter search
  evaluation.py          #   MAE/RMSE/bias and week-block bootstrap
  espn.py                #   ESPN league access and weekly projections
  predictions_log.py     #   append-only log of pre-game predictions
config/scoring.yaml      # full league scoring rules (offense, K, D/ST)
config/validation.yaml   # validation scheme (weekly walk-forward, tuning season)
config/model.yaml        # frozen hyperparameters and the search that chose them
notebooks/01_datos.ipynb
notebooks/02_features.ipynb
notebooks/03_backtest.ipynb
notebooks/04_predicciones.ipynb   # run weekly, before the games
data/predictions_log/    # versioned: one CSV per season
.env.example             # ESPN credentials template (the real .env is gitignored)
```

## Reproducing

```bash
git clone https://github.com/romogil99-max/fantasy-ml.git
cd fantasy-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # also installs src/fantasy_ml in editable mode

# Strip notebook outputs automatically on every commit (run once per clone)
nbstripout --install

# ESPN credentials (needed for a private league)
cp .env.example .env    # then fill in ESPN_LEAGUE_ID, ESPN_S2, ESPN_SWID, ESPN_TEAM_ID
```

`.env.example` explains where to find each value. Run the notebooks in order from the `notebooks/` directory. Downloaded data is cached in `data/`, which is gitignored except for `data/predictions_log/`.

Notebook 01 checks that `config/scoring.yaml` matches the league's live settings and stops if it doesn't. To use a different league, update that file first.

## Validation

**Weekly walk-forward:** for every week of 2024, 2025 and 2026 (so far), the model is trained only on games played before that week and then predicts it. Separate models are trained for QB/RB/WR/TE (position as a feature), K and D/ST.

| season | role | compared against |
|---|---|---|
| 2024 | hyperparameter tuning (in-sample for that choice) | rolling-average baseline |
| 2025 | out of sample | rolling-average baseline |
| 2026 | out of sample | baseline and ESPN projections |

The baseline is each player's average over their last 5 games. If that isn't available, it falls back to the season-to-date average, then last season's average, then the position average. The league was created in 2026, so ESPN projections exist only for 2026.

Metrics focus on **fantasy-relevant players**: each week, the top 24 QB, 48 RB, 60 WR, 24 TE, 20 K and 20 D/ST ranked by the baseline. Low-usage backups score close to 0, which would make every model look good. Differences between models come with 95% confidence intervals from a bootstrap that resamples whole weeks. With fewer than 6 weeks no interval is reported.

### How the hyperparameters were chosen

- **Search space:** a small grid of 12 combinations, `num_leaves` ∈ {7, 15, 31} × `min_child_samples` ∈ {20, 80} × `n_estimators` ∈ {200, 500}, with `learning_rate` 0.03 and fixed 80% row/column subsampling.
- **How each combination was evaluated:** with the weekly walk-forward over the **18 weeks of 2024 only** (training on 2023 plus earlier 2024 weeks), scored by MAE on fantasy-relevant players.
- **Result:** the best combination per model was **frozen** in `config/model.yaml` and used unchanged for 2025 and 2026. 2026 is never used for tuning; it has too few weeks.
- **What was chosen:** the simplest option won for all three models (7 leaves, 200 trees; min 80 samples per leaf for QB/RB/WR/TE and K, 20 for D/ST). Differences across the grid were small, e.g. 5.83 vs 5.99 MAE for QB/RB/WR/TE.

### Backtest results (fantasy-relevant players, MAE in points)

| position | 2025 model | 2025 baseline | improvement |
|---|---|---|---|
| QB | 6.59 | 7.31 | 10.0% |
| RB | 5.70 | 6.20 | 8.0% |
| WR | 5.79 | 6.14 | 5.8% |
| TE | 5.31 | 5.78 | 8.1% |
| K | 3.81 | 4.19 | 9.1% |
| D/ST | 4.86 | 5.42 | 10.3% |

- **2025 (out of sample):** the model beats the baseline at every position, and every 95% interval excludes zero. The gain matches 2024, the tuning season, so tuning did not inflate the results.
- **Against ESPN (2026 weeks 1–2):** these are the 292 player-weeks that were on a league roster, the only past ESPN projections still available. ESPN is slightly better, 6.25 vs 6.46 MAE. Two weeks is not conclusive; the predictions log will settle it.
- **Where it misses most:** 25+ point games (8% of games, 20% of the error, driven by touchdowns), players returning after 2+ weeks out (overestimated by about 2.4 points) and rookies in their first few games.

## Weekly predictions log

`notebooks/04_predicciones.ipynb` runs before each week's games. It trains on every game already played, predicts the upcoming week and appends to `data/predictions_log/<season>.csv`. Each row stores:

- the model prediction, the baseline and ESPN's projection;
- ESPN's injury status and whether the player is on my roster;
- the kickoff time, the generation timestamp and the git commit of the code.

The log is append-only and only records games that haven't started. It can be re-run (for example on Sunday morning after injury news); evaluation uses the last prediction made before each kickoff. The log is committed so git history timestamps each prediction. ESPN does not keep past weekly projections for every player, so this log is the only way to compare against ESPN over a full season.

## Limitations

- **Closing betting lines.** nflverse schedules provide closing spreads and totals. Real lineup decisions are made earlier, with opening or current lines, so backtests are somewhat optimistic.
- **The model assumes the player plays.** Targets exist only for games played. The model predicts points *if active*; availability (injuries, inactives) is handled outside the model, and the log keeps ESPN's injury status for that purpose.
- **K and D/ST scoring** is computed from nflverse stats and validated against ESPN's actual 2026 points (K 64/64 games, D/ST 63/64; the one mismatch is a sack credited differently by the two sources). The same rules are assumed for 2023–2025.
- **ESPN comparison in the backtest** covers only players on a league roster in 2026 weeks 1–2, because ESPN does not keep past projections for everyone else.

## Roadmap

- [x] **01_datos**: download weekly stats, connect to the league, validate scoring rules, and join the roster with ESPN projections through nflverse IDs
- [x] **02_features**: K and D/ST points validated against ESPN; rolling form, usage, expected points, game context and opponent features, with an automated test for future data leaking into them
- [x] **03_backtest**: weekly walk-forward for 2024–2026, hyperparameters tuned on 2024 only, error by position and failure analysis
- [x] **04_predicciones**: weekly pre-game predictions log with ESPN projections
- [ ] **05_lineup**: start/sit recommendations for my roster
