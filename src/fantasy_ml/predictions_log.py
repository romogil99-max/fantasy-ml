"""Registro de predicciones semanales: data/predictions_log/<temporada>.csv.

Reglas:
- Solo se agregan filas; nunca se modifican ni borran las anteriores.
- Solo se registran predicciones hechas ANTES del inicio de cada partido (pregame).
- Se puede ejecutar varias veces por semana (p. ej. el domingo con noticias de lesiones);
  al evaluar se usa la última predicción previa al inicio de cada partido.
- El archivo se versiona en git: los commits fechan cada predicción.
"""
import subprocess
from datetime import datetime, timezone

import polars as pl

from .data import PREDICTIONS_LOG, ROOT

LOG_SCHEMA = {
    "season": pl.Int32, "week": pl.Int32, "generated_at_utc": pl.Datetime("us", "UTC"), "code_version": pl.Utf8,
    "group": pl.Utf8, "entity_id": pl.Utf8, "espn_id": pl.Int64, "name": pl.Utf8, "position": pl.Utf8,
    "team": pl.Utf8, "opponent": pl.Utf8, "kickoff_utc": pl.Datetime("us", "UTC"),
    "pred_model": pl.Float64, "pred_baseline": pl.Float64, "espn_projection": pl.Float64,
    "espn_injury_status": pl.Utf8, "on_my_roster": pl.Boolean, "my_slot": pl.Utf8,
}
ENTITY = ["season", "week", "group", "entity_id"]


def log_path(season: int):
    return PREDICTIONS_LOG / f"{season}.csv"


def code_version() -> str:
    """Hash corto del commit actual; '-dirty' si hay cambios sin commit (sin contar el propio registro).

    Compara contenido con `git diff`, que aplica el filtro de nbstripout. `git status` no sirve aquí:
    marca como modificado un notebook ejecutado solo porque cambió de tamaño al tener resultados.
    """
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    scope = ["--", ".", ":!data/predictions_log"]
    sha = git("rev-parse", "--short", "HEAD").stdout.strip() or "unknown"
    changed = git("diff", "--quiet", "--no-textconv", "HEAD", *scope).returncode != 0
    untracked = git("ls-files", "--others", "--exclude-standard", *scope).stdout.strip()
    return f"{sha}-dirty" if changed or untracked else sha


def kickoff_utc(team_games: pl.DataFrame) -> pl.Expr:
    """gameday + gametime de nflverse (hora del Este de EE. UU.) en UTC."""
    return (pl.concat_str("gameday", pl.lit(" "), "gametime").str.to_datetime("%Y-%m-%d %H:%M")
              .dt.replace_time_zone("America/New_York").dt.convert_time_zone("UTC"))


def append(rows: pl.DataFrame, now: datetime | None = None) -> pl.DataFrame:
    """Agrega al registro las filas cuyo partido aún no empieza. Devuelve lo que se registró."""
    now = now or datetime.now(timezone.utc)
    rows = (rows.with_columns(generated_at_utc=pl.lit(now).cast(LOG_SCHEMA["generated_at_utc"]),
                              code_version=pl.lit(code_version()))
                .select([pl.col(c).cast(t) for c, t in LOG_SCHEMA.items()]))
    pregame = rows.filter(pl.col("kickoff_utc") > pl.col("generated_at_utc"))
    if pregame.is_empty():
        return pregame
    for season, part in pregame.partition_by("season", as_dict=True).items():
        path = log_path(season[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", encoding="utf-8") as f:
            part.write_csv(f, include_header=new_file)
    return pregame


def read(season: int) -> pl.DataFrame:
    path = log_path(season)
    if not path.exists():
        return pl.DataFrame(schema=LOG_SCHEMA)
    return pl.read_csv(path, schema=LOG_SCHEMA)


def latest_pregame(log: pl.DataFrame) -> pl.DataFrame:
    """La última predicción hecha antes del inicio de cada partido, por jugador/equipo y semana."""
    return (log.filter(pl.col("generated_at_utc") < pl.col("kickoff_utc"))
               .sort("generated_at_utc")
               .group_by(ENTITY).last())
