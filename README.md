# fantasy-ml

Predicting weekly fantasy football points for the players in my ESPN league (full PPR), and checking whether a model can beat ESPN's own projections.

> **Status: work in progress.** Data, features, a weekly walk-forward backtest, a weekly predictions log, a lineup helper and a trade analyzer (evaluation, uncertainty and trade finder) are done. The 2026 comparison against ESPN is accumulating week by week in `data/predictions_log/`.

## Stack

- **Python 3.11**, **Jupyter**
- [`nflreadpy`](https://github.com/nflverse/nflreadpy): weekly player stats, snaps, expected points, schedules and rosters from nflverse (2023–2026)
- [`espn-api`](https://github.com/cwendt94/espn-api): league settings, rosters and ESPN projections
- [`polars`](https://pola.rs/): data wrangling
- [`LightGBM`](https://lightgbm.readthedocs.io/): gradient-boosted trees
- [`Streamlit`](https://streamlit.io/) and [`Altair`](https://altair-viz.github.io/): dashboard
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
  lineup.py              #   optimal lineup from the league's slots, start/sit changes, replacement level
  trades.py              #   rest-of-season projections, availability model, trade values, Monte Carlo
  matchup.py             #   win probability against this week's opponent (also a CLI for the Sunday run)
  report.py              #   predictions log vs actual points (model and ESPN error)
  outlook.py             #   multi-week outlook: weekly points, byes and holes, pickups, upcoming opponents
app/dashboard.py         # Streamlit dashboard (probabilities, predictions, suggestions)
scripts/run_weekly_predictions.sh  # weekly automated run: notebook 04, Sunday win probability, commit + push
systemd/                 # user units: weekly predictions timer and dashboard service
config/scoring.yaml      # full league scoring rules (offense, K, D/ST)
config/validation.yaml   # validation scheme (weekly walk-forward, tuning season)
config/model.yaml        # frozen hyperparameters and the search that chose them
config/trades.yaml       # trade analyzer settings (playoff weight, injury status, availability, lines)
config/ranges.yaml       # per-position calibration of the P10–P90 ranges (fit on 2024, verified on 2025)
notebooks/01_datos.ipynb
notebooks/02_features.ipynb
notebooks/03_backtest.ipynb
notebooks/04_predicciones.ipynb   # run weekly, before the games (automated with a systemd timer)
notebooks/05_alineacion.ipynb     # lineup, P10–P90 ranges, free agents and win probability
notebooks/06_trades.ipynb         # trade analyzer
data/predictions_log/    # versioned: predictions (<season>.csv) and Sunday win probabilities (winprob_<season>.csv)
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

## Ranges and win probability

**P10–P90 ranges.** Two LightGBM quantile models (α = 0.1 and 0.9), with the same frozen hyperparameters and features, give each player a range; crossed quantiles are swapped, though none occurred in 15,615 backtest predictions. Raw ranges were too narrow for K and D/ST (~74% coverage). A per-position conformal adjustment was fit on 2024 and checked on 2025: coverage for fantasy-relevant players is **80–85% at every position**. The weekly predictions log stores each player's range before kickoff, so 2026 coverage can be measured too.

**Win probability against this week's opponent** (`matchup.py`, notebook 05):
- **Lineups:** mine is my current ESPN lineup, with the optimal one shown as an alternative. The opponent's is their current lineup, with empty slots and out/IR/doubtful/bye starters replaced by their best available bench player.
- **Simulation (10,000 runs):**
  - each player plays according to his ESPN status;
  - points are drawn from real out-of-sample residuals, scaled so each player's simulated P10–P90 matches his own range and centred on his prediction;
  - a starter who sits is replaced from the bench;
  - games already finished use actual points.
- **Multi-week outlook** (`outlook.py`): for the next 4 weeks it shows expected points with 80% ranges, win probability against each scheduled opponent, the bye calendar with real holes (slots my roster cannot fill), and free agents ranked by their gain over the horizon, including who to drop. Here free agents only fill holes, because it describes my own roster; in trade valuation they compete for every slot.
- **Close decisions:** for start/sit choices within 3 expected points, it shows the lower-variance option (right when favoured), the higher-variance option (right when not favoured) and the option with the higher win probability this week.
- **Sunday run:** the automated run appends the result to `winprob_<season>.csv`, alongside ESPN's own win probability.

## Dashboard

`app/dashboard.py` is a Streamlit app with one page and six tabs:

| Tab | What it shows |
|---|---|
| This week | Win probability with my current lineup, the optimal lineup and ESPN's; expected points with 80% ranges; the logged win-probability history; the opponent's lineup with holes filled |
| My lineup | Each player's prediction, P10–P90 range, ESPN projection, injury status and whether he starts in the optimal lineup, plus close start/sit decisions |
| Free agents | Top 5 per position, ranked by how much each one improves my optimal lineup |
| Upcoming weeks | Expected points per week with 80% ranges, win probability against each upcoming opponent, a bye calendar with real lineup holes, and free agents ranked by their gain over the horizon (2, 4 or 6 weeks) |
| Trades | Latest trade-finder results, with a button to run it again (~6 min) |
| How is the model doing? | 2026 error of the model vs ESPN by position, and real coverage of the P10–P90 ranges, from the predictions log |

ESPN data is fetched live and cached for 10 minutes; a sidebar button refreshes it. Predictions come from the predictions log, so they match what was committed before each game.

Run it once with `streamlit run app/dashboard.py`, or keep it running as a user service that starts at boot and restarts on failure:

```bash
export XDG_RUNTIME_DIR=/run/user/$(id -u)   # needed when the shell has no user session bus
ln -sf ~/fantasy-ml/systemd/fantasy-dashboard.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now fantasy-dashboard.service
# then open http://<server-ip>:8501
```

> ⚠ **The dashboard has no authentication.** The service listens on `0.0.0.0:8501`, so anyone on the local network can see the league, my roster and the other teams. Never forward that port on the router. To restrict it to the server itself, change `--server.address` to `127.0.0.1` in the service and use an SSH tunnel (`ssh -L 8501:localhost:8501 user@server`).

## Trade analyzer

`notebooks/06_trades.ipynb` values a trade for **both teams** as the change in expected points of each team's optimal lineup from now to week 17. Byes are included, and playoff weeks count double; the playoff weeks and the trade deadline are read from the league settings. Each player is projected week by week: current form features are frozen, each week's matchup context is swapped in, and betting lines not yet published are estimated from a team-strength model (1.2-point error on implied totals). A Monte Carlo over 2,000 seasons gives each estimate an 80% interval and a probability that the trade helps each team. It samples real errors from a horizon backtest (projection error grows from 5.7 to 6.6 points of SD between 1 and 10+ weeks ahead) and simulates availability as a Markov chain. ESPN's projections are shown alongside, as a proxy for how the other manager will see the trade.

The **trade finder** searches the other 9 rosters for 1-for-1 and 2-for-1 trades (both directions) where both teams gain. The search space has about 30,000 trades, so it runs in three steps. First, an additive screen built from marginal values (what each team loses by giving each player and gains by receiving each candidate, minus the player it would have to drop). This screen correlates 0.82–0.87 with the exact values. Second, an exact valuation of the ~800 best candidates, discarding redundant variants of simpler trades. Third, Monte Carlo and ESPN's view for the top 30.

## Findings

- **Individual injury history barely predicts future availability.** Only games missed while inactive or on a reserve list count; backup roles and week 18 rest are excluded. On that basis, each player's history (2023–2025) was blended with his position's rate, with the blend strength chosen by how well it predicted the next season. The best strength weighs the position rate like **~300 games**, and it improves on the position rate alone by **less than 0.1%**. Splitting positions by usage level did not help either (0.2%). What *does* matter is that absences come in streaks: a player who missed a game misses the next one 84% of the time. Short inactive stints end quickly (32% return the following week), while reserve/IR stints rarely do (96% stay out). So a healthy player's availability over the next few weeks is much higher than his season-long rate.
- **Replacement level changes the value of bench players a lot in a 10-team league.** With only 10 teams the waiver wire is deep: in 2026 week 3 the best free-agent QBs project 14–17 points. The best free agents at each position compete for every lineup slot, so a player is worth only what he adds over them. On top of that, a starter who may not play is backed up by his bench or a free agent. Once that is modelled, a backup's value as bye/injury insurance mostly disappears. In the example trade (my backup QB Brock Purdy for WR Nico Collins), the other team went from **+50.0** without replacement level to **−4.8** with it, while my side went from +16.9 to +23.9. For the trade I was actually considering (Purdy for RB TreVeyon Henderson), my side went from −21.5 to −1.1: a coin flip, with a 48% chance of helping me.

  Getting this right required letting free agents compete for *every* slot, not only empty ones. The first version filled only empty slots, so removing a starter who was worse than the best free agent made a team *more* valuable: +47.6 points for dropping one team's D/ST. The trade finder then suggested trades whose only "benefit" was getting rid of such players. The current rule is monotone: removing any of the 154 rostered players never increases a roster's value.
- **With replacement level, trades rarely move the needle.** Even the best trades the finder proposes are worth about 10 weighted points over 15 weeks for each side, under one point per week. Their probability of helping each team is only 50–57%. In this league the rosters are close to efficient given what the waiver wire offers.

## Limitations

- **Closing betting lines.** nflverse schedules provide closing spreads and totals. Real lineup decisions are made earlier, with opening or current lines, so backtests are somewhat optimistic.
- **The model assumes the player plays.** Targets exist only for games played. The model predicts points *if active*; availability (injuries, inactives) is handled outside the model, and the log keeps ESPN's injury status for that purpose.
- **K and D/ST scoring** is computed from nflverse stats and validated against ESPN's actual 2026 points (K 64/64 games, D/ST 63/64; the one mismatch is a sack credited differently by the two sources). The same rules are assumed for 2023–2025.
- **ESPN comparison in the backtest** covers only players on a league roster in 2026 weeks 1–2, because ESPN does not keep past projections for everyone else.
- **The dashboard has no authentication:** keep it on the local network only.
- **Win probability** treats players as independent. It ignores QB–receiver stacks and the negative link between a D/ST and the opposing offense, so probabilities are somewhat too extreme.
- **Trade analyzer:**
  - It assumes the best free agents (top 3 per position each week) are available whenever they beat a rostered player, and that every team can use them. That means unlimited streaming with no waiver competition. It is reasonable in a 10-team league, but optimistic.
  - It ignores the probability of making the playoffs, since playoff weeks are simply weighted ×2.
  - It ignores correlation between players on the same NFL team.
  - Future betting lines are estimated rather than observed.

## Roadmap

- [x] **01_datos**: download weekly stats, connect to the league, validate scoring rules, and join the roster with ESPN projections through nflverse IDs
- [x] **02_features**: K and D/ST points validated against ESPN; rolling form, usage, expected points, game context and opponent features, with an automated test for future data leaking into them
- [x] **03_backtest**: weekly walk-forward for 2024–2026, hyperparameters tuned on 2024 only, error by position and failure analysis
- [x] **04_predicciones**: weekly pre-game predictions log with ESPN projections
- [x] **05_alineacion**
  - [x] optimal lineup, start/sit changes flagged when inconclusive (< 3 points), top free agents by lineup gain
  - [x] calibrated P10–P90 ranges per player, logged before each game
  - [x] win probability against this week's opponent, with favourite/underdog picks for close decisions (automated on Sundays)
  - [x] multi-week outlook: weekly points with ranges, bye calendar and real lineup holes, free agents over the horizon, win probability against upcoming opponents
- [x] **Dashboard**: Streamlit app with win probability, lineup, free agents, trades and model tracking
- [x] **06_trades**
  - [x] (a) week-by-week rest-of-season projections and evaluation of a specific trade for both teams
  - [x] (b) uncertainty: horizon backtest and Monte Carlo
  - [x] (c) search for 1-for-1 and 2-for-1 trades where both teams gain
