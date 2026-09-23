"""Modelo (LightGBM), baseline y backtest walk-forward semana por semana."""
import itertools
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from .features import feature_cols

# Tres grupos con modelos separados: jugadores ofensivos, kickers y defensas
GROUPS = {
    "offense": {"baseline": ["fantasy_points_ppr_l5", "fantasy_points_ppr_std", "fantasy_points_ppr_prev"],
                "entity": "player_id"},
    "k": {"baseline": ["fantasy_points_l5", "fantasy_points_std", "fantasy_points_prev"], "entity": "player_id"},
    "dst": {"baseline": ["fantasy_points_l5", "fantasy_points_std", "fantasy_points_prev"], "entity": "team"},
}
POSITIONS = ["QB", "RB", "WR", "TE"]

# Parámetros comunes; los ajustables salen de config/model.yaml (elegidos solo con 2024)
FIXED_PARAMS = dict(objective="regression", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                    reg_lambda=1.0, random_state=0, n_jobs=4, verbose=-1)

# Relevantes para fantasy: los N primeros de cada posición y semana según el baseline
TOP_N = {"QB": 24, "RB": 48, "WR": 60, "TE": 24, "K": 20, "D/ST": 20}


def with_position(df: pl.DataFrame, group: str) -> pl.DataFrame:
    """Columna `position` homogénea en los tres grupos (K y D/ST no la traen)."""
    if group == "k":
        return df.with_columns(position=pl.lit("K"))
    if group == "dst":
        return df.with_columns(position=pl.lit("D/ST"))
    return df


def design_matrix(df: pl.DataFrame, group: str) -> tuple[np.ndarray, list[str]]:
    feats = feature_cols(df)
    X = df.select(feats)
    if group == "offense":  # posición en one-hot (LightGBM trabaja con numpy)
        X = X.with_columns([(df["position"] == p).cast(pl.Float64).alias(f"pos_{p}") for p in POSITIONS])
    X = X.cast(pl.Float64)
    return X.to_numpy(), X.columns


def fit(train: pl.DataFrame, group: str, params: dict) -> lgb.LGBMRegressor:
    X, names = design_matrix(train, group)
    model = lgb.LGBMRegressor(**FIXED_PARAMS, **params)
    model.fit(X, train["y"].to_numpy(), feature_name=names)
    return model


def predict(model: lgb.LGBMRegressor, df: pl.DataFrame, group: str) -> np.ndarray:
    X, _ = design_matrix(df, group)
    return model.booster_.predict(X)  # el booster no exige nombres de columnas en numpy


def baseline(df: pl.DataFrame, train: pl.DataFrame, group: str) -> pl.Series:
    """Media de los últimos 5 partidos; si no hay, la de la temporada o la anterior; si no, la media de la posición."""
    pos_mean = with_position(train, group).group_by("position").agg(pl.col("y").mean().alias("_pos_mean"))
    return (with_position(df, group)
            .join(pos_mean, on="position", how="left")
            .select(pl.coalesce(*GROUPS[group]["baseline"], "_pos_mean").alias("baseline"))["baseline"])


def before(season: int, week: int) -> pl.Expr:
    return (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))


def eval_weeks(df: pl.DataFrame, seasons) -> list[tuple[int, int]]:
    return (df.filter(pl.col("season").is_in(seasons), pl.col("y").is_not_null())
              .select("season", "week").unique().sort("season", "week").rows())


def walk_forward(df: pl.DataFrame, group: str, params: dict, weeks, verbose=False) -> pl.DataFrame:
    """Para cada semana: entrena con TODAS las filas anteriores y predice esa semana."""
    out = []
    for season, week in weeks:
        train = df.filter(before(season, week), pl.col("y").is_not_null())
        test = df.filter(pl.col("season") == season, pl.col("week") == week, pl.col("y").is_not_null())
        if test.is_empty():
            continue
        t0 = time.time()
        model = fit(train, group, params)
        keep = dict.fromkeys(["season", "week", GROUPS[group]["entity"], "player_display_name", "team", "opponent",
                              "opponent_team", "position", "y", "games_career", "games_season", "weeks_since_last",
                              "from_snaps_only", "espn_id"])
        test_pos = with_position(test, group)
        out.append(test_pos.select([c for c in keep if c in test_pos.columns])
                   .with_columns(pred=pl.Series(predict(model, test, group)),
                                 baseline=baseline(test, train, group),
                                 train_rows=pl.lit(train.height)))
        if verbose:
            print(f"  {group} {season} sem {week:>2}: train {train.height:>6,} · test {test.height:>4} · {time.time() - t0:.1f}s")
    return mark_relevant(pl.concat(out, how="diagonal_relaxed"))


def mark_relevant(pred: pl.DataFrame) -> pl.DataFrame:
    """relevant = entre los TOP_N de su posición y semana según el baseline (mismo grupo para todos los modelos)."""
    rank = pl.col("baseline").rank("ordinal", descending=True).over("season", "week", "position")
    return pred.with_columns(relevant=rank <= pl.col("position").replace_strict(TOP_N, return_dtype=pl.Int64))


def search(df: pl.DataFrame, group: str, grid: dict, season: int, verbose=True) -> pl.DataFrame:
    """Búsqueda en rejilla evaluada con el walk-forward semanal de UNA temporada (MAE en jugadores relevantes)."""
    weeks = eval_weeks(df, [season])
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        t0 = time.time()
        pred = walk_forward(df, group, params, weeks)
        rel = pred.filter("relevant")
        rows.append({**params,
                     "mae_relevant": (rel["pred"] - rel["y"]).abs().mean(),
                     "mae_all": (pred["pred"] - pred["y"]).abs().mean(),
                     "mae_baseline_relevant": (rel["baseline"] - rel["y"]).abs().mean(),
                     "seconds": round(time.time() - t0, 1)})
        if verbose:
            print(f"  {group} {params} → MAE relevantes {rows[-1]['mae_relevant']:.3f} ({rows[-1]['seconds']}s)")
    return pl.DataFrame(rows).sort("mae_relevant")


# ---------------------------------------------------------------- rangos (regresión por cuantiles)

QUANTILES = (0.1, 0.9)


def fit_quantile(train: pl.DataFrame, group: str, params: dict, alpha: float) -> lgb.LGBMRegressor:
    """LightGBM con pérdida de cuantil `alpha` y los mismos hiperparámetros congelados que el modelo de media."""
    X, names = design_matrix(train, group)
    model = lgb.LGBMRegressor(**{**FIXED_PARAMS, "objective": "quantile", "alpha": alpha}, **params)
    model.fit(X, train["y"].to_numpy(), feature_name=names)
    return model


def predict_range(models: tuple, df: pl.DataFrame, group: str, calibration: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """P10 y P90 de cada fila. Nunca se cruzan: si P10 > P90, se intercambian.

    `calibration` (opcional): {posición: (ajuste_inferior, ajuste_superior)} en puntos, estimado en otra
    temporada (calibrate_ranges): P10 − ajuste_inferior, P90 + ajuste_superior.
    """
    lo, hi = predict(models[0], df, group), predict(models[1], df, group)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    if calibration:
        pos = with_position(df, group)["position"].to_list()
        adj = np.array([calibration.get(p, (0.0, 0.0)) for p in pos])
        lo, hi = lo - adj[:, 0], hi + adj[:, 1]
    return lo, hi


def walk_forward_ranges(df: pl.DataFrame, group: str, params: dict, weeks, verbose=False) -> pl.DataFrame:
    """Walk-forward semanal de los cuantiles: para cada semana entrena con las anteriores y predice P10/P90.

    Incluye cuántas veces se cruzaron los dos modelos antes de ordenarlos (`crossed`).
    """
    out = []
    for season, week in weeks:
        train = df.filter(before(season, week), pl.col("y").is_not_null())
        test = df.filter(pl.col("season") == season, pl.col("week") == week, pl.col("y").is_not_null())
        if test.is_empty():
            continue
        models = tuple(fit_quantile(train, group, params, a) for a in QUANTILES)
        raw_lo, raw_hi = predict(models[0], test, group), predict(models[1], test, group)
        lo, hi = predict_range(models, test, group)
        keep = [c for c in dict.fromkeys(["season", "week", GROUPS[group]["entity"], "player_display_name", "team", "position", "y"])
                if c in with_position(test, group).columns]
        out.append(with_position(test, group).select(keep)
                   .with_columns(q10=pl.Series(lo), q90=pl.Series(hi), crossed=pl.Series(raw_lo > raw_hi)))
        if verbose:
            print(f"  {group} {season} sem {week:>2}")
    return pl.concat(out, how="diagonal_relaxed")


def range_coverage(r: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Cobertura del rango: % dentro [P10, P90] (objetivo 80), % por debajo y por encima (objetivo 10 cada uno)."""
    return (r.group_by(by)
             .agg(pl.len().alias("n"),
                  ((pl.col("y") >= pl.col("q10")) & (pl.col("y") <= pl.col("q90"))).mean().mul(100).round(1).alias("pct_dentro"),
                  (pl.col("y") < pl.col("q10")).mean().mul(100).round(1).alias("pct_debajo"),
                  (pl.col("y") > pl.col("q90")).mean().mul(100).round(1).alias("pct_encima"),
                  (pl.col("q90") - pl.col("q10")).mean().round(2).alias("ancho_medio"))
             .sort(by))


def calibrate_ranges(r: pl.DataFrame, target_tail: float = 0.10) -> dict:
    """Ajuste por posición para que cada cola tenga `target_tail` (calibración conformal, estilo CQR).

    ajuste_inferior = cuantil (1 − target_tail) de (P10 − y); ajuste_superior = ídem de (y − P90). Si el
    rango ya es demasiado ancho, el ajuste sale negativo y lo estrecha. Se estima en una temporada y
    se aplica a otra.
    """
    cal = {}
    for (pos,), g in r.partition_by("position", as_dict=True).items():
        lo = float(np.quantile((g["q10"] - g["y"]).to_numpy(), 1 - target_tail))
        hi = float(np.quantile((g["y"] - g["q90"]).to_numpy(), 1 - target_tail))
        cal[pos] = (round(lo, 3), round(hi, 3))
    return cal
