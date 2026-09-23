"""Evaluación del registro de predicciones contra los resultados reales (para el dashboard y el notebook 04)."""
import polars as pl

from . import data, features as F, predictions_log as plog, scoring
from .data import KEYS


def actual_points(season: int, refresh: bool = False) -> pl.DataFrame:
    """Puntos reales por jugador/equipo y semana: nflverse (QB/RB/WR/TE) y cálculo validado de K y D/ST."""
    src = data.load_sources(refresh=refresh)
    sc = data.load_config("scoring")
    base = F.offense_base(src["player_stats"], src["snap_counts"], src["opportunity"], src["playerids"])
    return pl.concat([
        base.filter(pl.col("season") == season).select(*KEYS, group=pl.lit("offense"), entity_id="player_id", y="fantasy_points_ppr"),
        scoring.k_points(src["player_stats"], sc).filter(pl.col("season") == season)
               .select(*KEYS, group=pl.lit("k"), entity_id="player_id", y="fantasy_points"),
        scoring.dst_points(src["team_stats"], src["schedules"], sc).filter(pl.col("season") == season)
               .select(*KEYS, group=pl.lit("dst"), entity_id="team", y="fantasy_points"),
    ])


def logged_vs_actual(season: int, refresh: bool = False) -> pl.DataFrame:
    """Última predicción previa al partido de cada jugador, cruzada con sus puntos reales (solo si jugó)."""
    log = plog.latest_pregame(plog.read(season))
    return log.join(actual_points(season, refresh), on=["season", "week", "group", "entity_id"], how="inner")
