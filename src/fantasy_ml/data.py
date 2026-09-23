"""Rutas del proyecto, configuración y descarga de datos de nflverse con caché local."""
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
PREDICTIONS_LOG = ROOT / "data" / "predictions_log"
CONFIG = ROOT / "config"

SEASONS = [2023, 2024, 2025, 2026]
KEYS = ["season", "week"]


def load_config(name: str) -> dict:
    return yaml.safe_load((CONFIG / f"{name}.yaml").read_text(encoding="utf-8"))


def cached(name: str, loader, refresh: bool = False) -> pl.DataFrame:
    """Lee data/raw/<name>.parquet o lo descarga con `loader` si no existe (o si refresh)."""
    path = DATA_RAW / f"{name}.parquet"
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df = loader()
    df.write_parquet(path)
    return df


def load_sources(seasons=SEASONS, refresh: bool = False) -> dict[str, pl.DataFrame]:
    """Todas las fuentes de nflverse que usa el proyecto. 2026 está en curso: refresh=True cada semana."""
    loaders = {
        "player_stats": ("player_stats_weekly_2023_2026", lambda: nfl.load_player_stats(seasons, summary_level="week")),
        "team_stats": ("team_stats_weekly_2023_2026", lambda: nfl.load_team_stats(seasons, summary_level="week")),
        "schedules": ("schedules_2023_2026", lambda: nfl.load_schedules(seasons)),
        "snap_counts": ("snap_counts_2023_2026", lambda: nfl.load_snap_counts(seasons)),
        "opportunity": ("ff_opportunity_2023_2026", lambda: nfl.load_ff_opportunity(seasons)),
        "playerids": ("ff_playerids", nfl.load_ff_playerids),
    }
    return {key: cached(name, loader, refresh) for key, (name, loader) in loaders.items()}


def load_rosters_weekly(season: int, refresh: bool = False) -> pl.DataFrame:
    """Rosters semanales de nflverse (equipo y estado ACT/INA/RES de cada jugador por semana)."""
    return cached(f"rosters_weekly_{season}", lambda: nfl.load_rosters_weekly([season]), refresh)
