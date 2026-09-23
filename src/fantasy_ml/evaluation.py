"""Métricas de error y comparación entre modelos con intervalos bootstrap por semana."""
import numpy as np
import polars as pl


def metrics(pred: pl.DataFrame, by: list[str], models=("pred", "baseline")) -> pl.DataFrame:
    """MAE, RMSE y sesgo (predicho - real) de cada modelo, agrupado por `by`."""
    aggs = [pl.len().alias("n")]
    for m in models:
        err = pl.col(m) - pl.col("y")
        aggs += [err.abs().mean().round(3).alias(f"mae_{m}"),
                 (err ** 2).mean().sqrt().round(3).alias(f"rmse_{m}"),
                 err.mean().round(3).alias(f"bias_{m}")]
    return pred.group_by(by).agg(aggs).sort(by)


MIN_WEEKS = 6  # con menos semanas el bootstrap por semana no da un intervalo creíble


def bootstrap_mae_diff(pred: pl.DataFrame, a="pred", b="baseline", n_boot=2000, seed=0) -> dict:
    """Diferencia de MAE (a - b) con IC 95% remuestreando SEMANAS completas (los errores de una semana están correlacionados).

    Con menos de MIN_WEEKS semanas el intervalo no es fiable: úsese solo el valor puntual.
    """
    wk = (pred.with_columns(ea=(pl.col(a) - pl.col("y")).abs(), eb=(pl.col(b) - pl.col("y")).abs())
              .group_by("season", "week").agg(pl.col("ea").sum(), pl.col("eb").sum(), pl.len().alias("n")))
    ea, eb, n = wk["ea"].to_numpy(), wk["eb"].to_numpy(), wk["n"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wk), size=(n_boot, len(wk)))
    diffs = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
    point = (ea.sum() - eb.sum()) / n.sum()
    return {"mae_diff": round(float(point), 3), "ci_low": round(float(np.percentile(diffs, 2.5)), 3),
            "ci_high": round(float(np.percentile(diffs, 97.5)), 3), "weeks": len(wk), "n": int(n.sum())}


def verdict(ci_low: pl.Expr, ci_high: pl.Expr, weeks: pl.Expr, better="modelo mejor", worse="referencia mejor") -> pl.Expr:
    return (pl.when(weeks < MIN_WEEKS).then(pl.lit(f"pocas semanas (<{MIN_WEEKS}): sin conclusión"))
              .when(ci_high < 0).then(pl.lit(better))
              .when(ci_low > 0).then(pl.lit(worse))
              .otherwise(pl.lit("sin diferencia clara")))
