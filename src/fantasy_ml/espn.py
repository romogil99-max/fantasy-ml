"""Acceso a la liga de ESPN. Las credenciales se leen del .env y nunca se imprimen."""
import os

import polars as pl
from dotenv import load_dotenv
from espn_api.football import League

from .data import ROOT

FA_POSITIONS = ["QB", "RB", "WR", "TE", "K", "D/ST"]


def connect(year: int = 2026) -> League:
    load_dotenv(ROOT / ".env")
    missing = [k for k in ("ESPN_LEAGUE_ID", "ESPN_S2", "ESPN_SWID", "ESPN_TEAM_ID") if not os.getenv(k)]
    if missing:
        raise EnvironmentError(f"Faltan variables en .env: {missing}. Usa .env.example como plantilla.")
    return League(league_id=int(os.environ["ESPN_LEAGUE_ID"]), year=year,
                  espn_s2=os.environ["ESPN_S2"], swid=os.environ["ESPN_SWID"])


def my_team_id() -> int:
    load_dotenv(ROOT / ".env")
    return int(os.environ["ESPN_TEAM_ID"])


def rostered_projections(league: League, week: int) -> pl.DataFrame:
    """Proyección de ESPN para la semana de todos los jugadores con roster en la liga (desde los box scores).

    Es la única forma de recuperar proyecciones de semanas pasadas: ESPN no las guarda para el resto.
    """
    me = my_team_id()
    rows = []
    for box in league.box_scores(week):
        for team, lineup in ((box.home_team, box.home_lineup), (box.away_team, box.away_lineup)):
            team_id = getattr(team, "team_id", None)
            for p in lineup:
                rows.append({"espn_id": p.playerId, "espn_name": p.name, "espn_position": p.position,
                             "espn_projection": p.projected_points, "espn_injury_status": p.injuryStatus,
                             "fantasy_team_id": team_id, "on_my_roster": team_id == me, "my_slot": p.slot_position if team_id == me else None})
    return _frame(rows)


def free_agent_projections(league: League, week: int, size: int = 150) -> pl.DataFrame:
    """Proyección de ESPN para la semana de los agentes libres, por posición (los más rostereados primero)."""
    rows = []
    for pos in FA_POSITIONS:
        for p in league.free_agents(week=week, size=size, position=pos):
            rows.append({"espn_id": p.playerId, "espn_name": p.name, "espn_position": p.position,
                         "espn_projection": p.projected_points, "espn_injury_status": p.injuryStatus,
                         "fantasy_team_id": None, "on_my_roster": False, "my_slot": None})
    return _frame(rows)


def week_projections(league: League, week: int) -> pl.DataFrame:
    """Rostereados + agentes libres, un registro por jugador."""
    return pl.concat([rostered_projections(league, week), free_agent_projections(league, week)]).unique("espn_id", keep="first")


def _frame(rows) -> pl.DataFrame:
    # ESPN devuelve [] como estado de lesión de los D/ST: se normaliza a nulo
    for r in rows:
        if not isinstance(r["espn_injury_status"], str):
            r["espn_injury_status"] = None
    return pl.DataFrame(rows, schema={"espn_id": pl.Int64, "espn_name": pl.Utf8, "espn_position": pl.Utf8,
                                      "espn_projection": pl.Float64, "espn_injury_status": pl.Utf8,
                                      "fantasy_team_id": pl.Int64, "on_my_roster": pl.Boolean, "my_slot": pl.Utf8})
