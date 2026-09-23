"""Analizador de trades: proyección semana a semana del resto de la temporada y valor de un trade.

Valor de un roster = suma ponderada, semana a semana, de los puntos esperados de su alineación óptima
(lineup.py), con las semanas de playoffs de la liga ponderadas por `playoff_weight`. Las semanas libres
(bye) valen 0 y el jugador no puede ser titular. El valor de un trade para un equipo es el cambio en
ese valor, y se calcula para los dos equipos.

Proyección de una semana futura: features de forma congeladas en su valor actual (lo último que se
sabe) y contexto del partido de esa semana (rival, local/visitante, descanso, estadio, líneas de
apuestas o su estimación si aún no se publican, y lo que permite hoy la defensa rival). Se usan los
mismos modelos que en las predicciones semanales (hiperparámetros congelados de config/model.yaml).
"""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import polars as pl

from . import features as F, lineup as L, model as M, scoring
from .data import KEYS

# defaultPositionId de ESPN → posición (para los límites por posición del roster)
ESPN_POSITION_IDS = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "D/ST"}
IR_STATUSES = {"INJURY_RESERVE", "IR"}
# Cada puntaje = proyección si juega x probabilidad de jugar (columna de probabilidad asociada)
SCORE_PROB = {"exp_points": "p_play", "espn_points": "p_espn"}
STATE_STATUSES = ("ACT", "RES", "INA", "EXE")  # estados de nflverse con los que se construye el estado actual (EXE = exento)


# ---------------------------------------------------------------- liga

@dataclass
class LeagueCalendar:
    season: int
    current_week: int
    reg_weeks: list[int]
    playoff_weeks: list[int]
    trade_deadline: datetime
    playoff_weight: float

    @property
    def weeks(self) -> list[int]:
        """Semanas que quedan (incluida la actual, si no ha empezado)."""
        return [w for w in self.reg_weeks + self.playoff_weeks if w >= self.current_week]

    def weight(self, week: int) -> float:
        return self.playoff_weight if week in self.playoff_weeks else 1.0

    def weights(self) -> pl.DataFrame:
        return pl.DataFrame({"week": self.weeks, "weight": [self.weight(w) for w in self.weeks],
                             "phase": ["playoffs" if w in self.playoff_weeks else "regular" for w in self.weeks]},
                            schema_overrides={"week": pl.Int32})


def league_calendar(league, playoff_weight: float) -> LeagueCalendar:
    """Semanas de temporada regular y de playoffs y fecha límite, leídas de la configuración de la liga."""
    s = league.settings
    periods = {int(k): v for k, v in s.matchup_periods.items()}
    reg = [w for p in sorted(periods) if p <= s.reg_season_count for w in periods[p]]
    playoffs = [w for p in sorted(periods) if p > s.reg_season_count for w in periods[p] if w <= league.finalScoringPeriod]
    return LeagueCalendar(season=league.year, current_week=league.current_week, reg_weeks=reg, playoff_weeks=playoffs,
                          trade_deadline=datetime.fromtimestamp(s.trade_deadline / 1000, tz=timezone.utc),
                          playoff_weight=playoff_weight)


def league_rules(league) -> dict:
    """Slots titulares y límites por posición del roster."""
    raw = league.espn_request.get_league()["settings"]["rosterSettings"]["positionLimits"]
    limits = {ESPN_POSITION_IDS[int(k)]: v for k, v in raw.items() if int(k) in ESPN_POSITION_IDS and v > 0}
    return {"slots": {k: v for k, v in league.settings.position_slot_counts.items() if v}, "position_limits": limits}


def league_rosters(league) -> pl.DataFrame:
    """Un registro por jugador con roster en la liga, con la media por partido que proyecta ESPN."""
    rows = []
    for t in league.teams:
        for p in t.roster:
            status = p.injuryStatus if isinstance(p.injuryStatus, str) else None
            rows.append({"fantasy_team_id": t.team_id, "fantasy_team": t.team_name, "espn_id": p.playerId,
                         "name": p.name, "position": p.position, "pro_team": F.ESPN_TO_NFLVERSE.get(p.proTeam, p.proTeam),
                         "lineup_slot": p.lineupSlot, "injury_status": status,
                         "espn_avg_proj": float(p.projected_avg_points or 0.0)})
    return pl.DataFrame(rows)


def league_free_agents(league, week: int, size: int = 100) -> pl.DataFrame:
    """Agentes libres de ESPN (mismas columnas que league_rosters, sin equipo de fantasy)."""
    rows = []
    for pos in ["QB", "RB", "WR", "TE", "K", "D/ST"]:
        for p in league.free_agents(week=week, size=size, position=pos):
            status = p.injuryStatus if isinstance(p.injuryStatus, str) else None
            rows.append({"fantasy_team_id": None, "fantasy_team": None, "espn_id": p.playerId, "name": p.name,
                         "position": p.position, "pro_team": F.ESPN_TO_NFLVERSE.get(p.proTeam, p.proTeam),
                         "lineup_slot": None, "injury_status": status, "espn_avg_proj": float(p.projected_avg_points or 0.0)})
    return pl.DataFrame(rows, schema_overrides={"fantasy_team_id": pl.Int64, "fantasy_team": pl.Utf8, "lineup_slot": pl.Utf8}).unique("espn_id")


# ---------------------------------------------------------------- disponibilidad

PLAYED, MISSED, ROLE = "jugó", "perdido", "rol"


def availability_history(rosters_weekly: pl.DataFrame, base: pl.DataFrame, points_k: pl.DataFrame,
                         team_games: pl.DataFrame, max_week: int = 17) -> pl.DataFrame:
    """Una fila por jugador y partido de su equipo (temporada regular, semanas <= max_week).

    outcome:
    - "jugó": tiene registro de snaps/estadísticas ese partido.
    - "perdido": estaba inactivo (INA) o en reserva/lesionados (RES) y no jugó → lesión o inactividad.
    - "rol": estaba activo (ACT) pero no jugó (suplente, banqueado). No cuenta como disponible ni
      como perdido: es cuestión de rol, no de salud. Se excluye del cálculo de disponibilidad.
    La semana 18 se excluye (descansos de fin de temporada; la liga termina en la 17). Solo cuentan
    semanas en que su equipo jugó (sin byes) y en que el jugador estaba en el equipo (no cortado).
    """
    games = team_games.filter(pl.col("week") <= max_week).select("season", "week", "team").unique()
    weeks = (rosters_weekly
        .filter(pl.col("game_type") == "REG", pl.col("week") <= max_week, pl.col("gsis_id").is_not_null(),
                pl.col("position").is_in(F.POS + ["K"]), pl.col("status").is_in(["ACT", "INA", "RES"]))
        .select(*F.INT_KEYS, player_id="gsis_id", position="position", team="team", status="status")
        .unique(["season", "week", "player_id"], keep="first")
        .join(games, on=["season", "week", "team"]))
    played = pl.concat([base.select("season", "week", "player_id"), points_k.select("season", "week", "player_id")]).unique()
    return (weeks.join(played.with_columns(_p=pl.lit(True)), on=["season", "week", "player_id"], how="left")
                 .with_columns(outcome=pl.when(pl.col("_p")).then(pl.lit(PLAYED))
                                         .when(pl.col("status").is_in(["INA", "RES"])).then(pl.lit(MISSED))
                                         .otherwise(pl.lit(ROLE)))
                 .drop("_p"))


def availability_rates(history: pl.DataFrame, base: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """Tasa base de partidos jugados (jugados / (jugados + perdidos)) por posición.

    Se calcula con jugadores relevantes (xfp medio >= min_xfp en las semanas 1-4 de la temporada) y sus
    partidos desde la semana 5. K: kickers con al menos 2 partidos en las semanas 1-4. D/ST: 1.
    Es la referencia (prior) hacia la que se contrae el historial de cada jugador. Separar además por
    nivel de uso (xfp) no mejoró la predicción de forma apreciable (notebook 06), así que no se hace.
    """
    seasons, min_xfp = cfg["seasons"], cfg["min_xfp"]
    h = history.filter(pl.col("season").is_in(seasons), pl.col("outcome") != ROLE)
    relevant = (base.filter(pl.col("season").is_in(seasons), pl.col("week") <= 4)
                    .group_by("season", "player_id").agg(pl.col("position").last(), pl.col("xfp").mean())
                    .filter(pl.col("xfp") >= min_xfp).select("season", "player_id", "position"))
    k_early = h.filter(pl.col("position") == "K", pl.col("week") <= 4, pl.col("outcome") == PLAYED).group_by("season", "player_id").len()
    relevant = pl.concat([relevant, k_early.filter(pl.col("len") >= 2).select("season", "player_id", position=pl.lit("K"))])
    rates = (h.filter(pl.col("week") >= 5).drop("position").join(relevant, on=["season", "player_id"])
              .group_by("position").agg(pl.col("season").n_unique().alias("seasons"),
                                        pl.struct("season", "player_id").n_unique().alias("player_seasons"),
                                        pl.len().alias("games"), (pl.col("outcome") == PLAYED).mean().alias("play_rate")))
    dst = pl.DataFrame({"position": ["D/ST"], "seasons": [None], "player_seasons": [None], "games": [None], "play_rate": [1.0]})
    return pl.concat([rates, dst], how="vertical_relaxed").sort("position")


def prior_play_rate(df: pl.DataFrame, rates: pl.DataFrame) -> pl.Series:
    """Tasa base de la posición de cada jugador."""
    return df.join(rates.select("position", prior_rate="play_rate"), on="position", how="left")["prior_rate"]


def shrinkage_strength(history: pl.DataFrame, base: pl.DataFrame, rates: pl.DataFrame, cfg: dict,
                       grid=(5, 10, 20, 40, 80, 120, 200, 300, 500, 1000)) -> tuple[float, pl.DataFrame]:
    """Peso κ (en partidos) de la tasa base frente al historial del jugador, elegido por capacidad predictiva.

    Para cada temporada T (desde la segunda de `seasons`): con el historial de las temporadas anteriores
    se predice la disponibilidad de los jugadores relevantes en T (semanas 5-17) y se mide el error
    cuadrático ponderado por partidos. Se elige el κ con menor error. Se evita el método de momentos
    porque las lesiones vienen en rachas (una rotura = muchos partidos perdidos seguidos) y eso infla la
    diferencia aparente entre jugadores.
    Devuelve (κ, tabla de error por κ).
    """
    seasons, min_xfp = cfg["seasons"], cfg["min_xfp"]
    h = history.filter(pl.col("outcome") != ROLE)
    rows = []
    for target in seasons[1:]:
        rel = (base.filter(pl.col("season") == target, pl.col("week") <= 4)
                   .group_by("player_id").agg(pl.col("position").last(), pl.col("xfp").mean())
                   .filter(pl.col("xfp") >= min_xfp))
        actual = (h.filter(pl.col("season") == target, pl.col("week") >= 5)
                   .group_by("player_id").agg((pl.col("outcome") == PLAYED).mean().alias("actual"), pl.len().alias("n_target")))
        past = (h.filter(pl.col("season") < target)
                 .group_by("player_id").agg((pl.col("outcome") == PLAYED).sum().alias("p"), (pl.col("outcome") == MISSED).sum().alias("m")))
        d = rel.join(actual, on="player_id").join(past, on="player_id", how="left").with_columns(pl.col("p", "m").fill_null(0))
        d = d.with_columns(prior_play_rate(d, rates))
        for k in grid:
            pred = (d["p"] + k * d["prior_rate"]) / (d["p"] + d["m"] + k)
            rows.append({"season": target, "kappa": k, "sse": float(((pred - d["actual"]) ** 2 * d["n_target"]).sum()),
                         "games": int(d["n_target"].sum())})
    res = (pl.DataFrame(rows).group_by("kappa").agg((pl.col("sse").sum() / pl.col("games").sum()).sqrt().alias("rmse"))
             .sort("kappa"))
    return float(res.sort("rmse")["kappa"][0]), res


def player_play_rate(df: pl.DataFrame, history: pl.DataFrame, rates: pl.DataFrame, kappa: float) -> pl.DataFrame:
    """Tasa de partidos jugados de cada jugador: su historial contraído hacia la tasa de su posición y nivel.

    rate = (jugados + κ·tasa_base) / (jugados + perdidos + κ). Sin historial, rate = tasa_base.
    Necesita las columnas entity_id (gsis_id; equipo en D/ST) y position.
    """
    own = (history.filter(pl.col("outcome") != ROLE)
                  .group_by("player_id").agg((pl.col("outcome") == PLAYED).sum().alias("hist_played"),
                                             (pl.col("outcome") == MISSED).sum().alias("hist_missed")))
    df = (df.with_columns(prior_play_rate(df, rates))
            .join(own.rename({"player_id": "entity_id"}), on="entity_id", how="left")
            .with_columns(pl.col("hist_played", "hist_missed").fill_null(0)))
    return df.with_columns(play_rate=pl.when(pl.col("position") == "D/ST").then(1.0).otherwise(
        (pl.col("hist_played") + kappa * pl.col("prior_rate")) / (pl.col("hist_played") + pl.col("hist_missed") + kappa)))


AVAIL_STATES = ["P", "INA", "RES"]  # jugando, ausencia corta (inactivo), ausencia larga (reserva/IR)


def absence_transitions(history: pl.DataFrame, seasons=None) -> np.ndarray:
    """Matriz de transición semanal 3x3 entre jugando (P), inactivo (INA) y reserva/IR (RES).

    Estimada con partidos consecutivos del mismo jugador en la misma temporada (sin contar los de rol).
    INA suele ser una ausencia corta y RES una larga: por eso se separan.
    """
    h = history.filter(pl.col("outcome") != ROLE)
    if seasons:
        h = h.filter(pl.col("season").is_in(seasons))
    h = (h.sort("player_id", "season", "week")
          .with_columns(state=pl.when(pl.col("outcome") == PLAYED).then(pl.lit("P")).otherwise(pl.col("status")))
          .with_columns(nxt=pl.col("state").shift(-1).over("player_id", "season"))
          .filter(pl.col("nxt").is_not_null()))
    T = np.zeros((3, 3))
    for a, b, n in h.group_by("state", "nxt").len().rows():
        T[AVAIL_STATES.index(a), AVAIL_STATES.index(b)] += n
    return T / T.sum(axis=1, keepdims=True)


def _stationary(T: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(T.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    return pi / pi.sum()


def player_chain(T: np.ndarray, play_rate: float) -> np.ndarray:
    """Cadena del jugador: se escalan las transiciones que salen de "jugando" para que la fracción de
    semanas jugando a largo plazo sea su `play_rate` (duración de las ausencias: la del historial)."""
    target = min(max(play_rate, 0.05), 0.999)
    lo, hi = 0.0, 1.0 / max(T[0, 1] + T[0, 2], 1e-9)
    for _ in range(50):
        c = (lo + hi) / 2
        Tc = T.copy()
        Tc[0, 1:] = T[0, 1:] * c
        Tc[0, 0] = 1 - Tc[0, 1:].sum()
        if _stationary(Tc)[0] > target:
            lo = c
        else:
            hi = c
    return Tc


def initial_state(injury_status, on_ir: bool, cfg: dict) -> np.ndarray:
    """Distribución (P, INA, RES) en la semana actual según el estado de ESPN."""
    if on_ir:
        return np.array([0.0, 0.0, 1.0])
    p = cfg["status_play_prob"].get(injury_status, 1.0) if isinstance(injury_status, str) else 1.0
    return np.array([p, 1 - p, 0.0])


def availability_paths(players: pl.DataFrame, cal: LeagueCalendar, cfg: dict, T: np.ndarray) -> pl.DataFrame:
    """Probabilidad de jugar de cada jugador en cada semana restante (marginal exacta de su cadena).

    Semana actual: estado de ESPN. IR: ausencia forzada `ir_weeks` semanas y después sigue desde RES.
    El tiempo avanza también en los byes (la recuperación sigue), pero en bye no juega. D/ST: 1.
    `players`: una fila por jugador con espn_id, position, play_rate, injury_status, lineup_slot, nfl_status.
    """
    weeks, now = cal.weeks, cal.current_week
    rows = []
    for r in players.to_dicts():
        if r["position"] == "D/ST":
            rows += [{"espn_id": r["espn_id"], "week": w, "p_avail": 1.0} for w in weeks]
            continue
        on_ir = (r["injury_status"] in IR_STATUSES) or (r["lineup_slot"] == "IR") or (r["nfl_status"] == "RES")
        Ti = player_chain(T, r["play_rate"] if r["play_rate"] is not None else 1.0)
        dist = initial_state(r["injury_status"], on_ir, cfg)
        for k, w in enumerate(weeks):
            if k > 0:
                dist = dist @ Ti
            if on_ir and w < now + cfg["ir_weeks"]:
                dist = np.array([0.0, 0.0, 1.0])
            rows.append({"espn_id": r["espn_id"], "week": w, "p_avail": float(dist[0])})
    return pl.DataFrame(rows, schema={"espn_id": pl.Int64, "week": pl.Int32, "p_avail": pl.Float64})


def add_play_rates(proj: pl.DataFrame, history: pl.DataFrame, rates: pl.DataFrame, kappa: float) -> pl.DataFrame:
    """Añade a la proyección la tasa de partidos jugados de cada jugador (constante en todas las semanas)."""
    ents = proj.select("entity_id", "position").unique("entity_id")
    pr = player_play_rate(ents, history, rates, kappa).select("entity_id", "prior_rate", "hist_played", "hist_missed", "play_rate")
    return proj.join(pr, on="entity_id", how="left")


# ---------------------------------------------------------------- líneas de apuestas futuras

def line_model(lines: pl.DataFrame, season: int, through_week: int, cfg: dict):
    """Ajusta implied_team ≈ μ + h·local + ataque[equipo] + defensa[rival] con las líneas ya publicadas.

    `lines`: una fila por equipo y partido con team, opponent, is_home e implied_team (nulo si no hay línea).
    Usa la temporada actual hasta `through_week` y la anterior con peso `prev_season_weight`.
    Devuelve una función que estima (implied_team, implied_opp) para cualquier partido.
    """
    obs = lines.filter(pl.col("implied_team").is_not_null(),
                       ((pl.col("season") == season) & (pl.col("week") <= through_week)) | (pl.col("season") == season - 1))
    teams = sorted(set(lines.filter(pl.col("season").is_in([season, season - 1]))["team"]))
    idx, nt = {t: i for i, t in enumerate(teams)}, len(teams)
    X = np.zeros((obs.height, 2 + 2 * nt))
    X[:, 0] = 1.0
    X[:, 1] = obs["is_home"].to_numpy()
    for r, (t, o) in enumerate(zip(obs["team"], obs["opponent"])):
        X[r, 2 + idx[t]] = 1.0
        X[r, 2 + nt + idx[o]] = 1.0
    y = obs["implied_team"].to_numpy()
    w = np.where(obs["season"].to_numpy() == season, 1.0, cfg["prev_season_weight"])
    reg = np.eye(X.shape[1]) * cfg["ridge_lambda"]
    reg[0, 0] = reg[1, 1] = 0.0  # sin penalizar la media ni la ventaja de local
    beta = np.linalg.solve(X.T @ (X * w[:, None]) + reg, X.T @ (w * y))

    def predict(team: str, opponent: str, is_home: int) -> tuple[float, float]:
        att = lambda t: beta[2 + idx[t]] if t in idx else 0.0
        dfn = lambda t: beta[2 + nt + idx[t]] if t in idx else 0.0
        mine = beta[0] + beta[1] * is_home + att(team) + dfn(opponent)
        theirs = beta[0] + beta[1] * (1 - is_home) + att(opponent) + dfn(team)
        return float(mine), float(theirs)

    return predict


def future_context(context: pl.DataFrame, team_games: pl.DataFrame, season: int, weeks: list[int],
                   through_week: int, cfg: dict) -> pl.DataFrame:
    """Contexto de cada equipo y semana futura; líneas reales si existen, estimadas si no.

    `lines_estimated` indica cuáles son estimaciones. El clima de partidos futuros al aire libre queda nulo.
    """
    lines = context.join(team_games.select(*KEYS, "team", "opponent"), on=[*KEYS, "team"])
    predict = line_model(lines, season, through_week, cfg)
    ctx = lines.filter(pl.col("season") == season, pl.col("week").is_in(weeks))
    est = [predict(t, o, h) for t, o, h in zip(ctx["team"], ctx["opponent"], ctx["is_home"])]
    it = pl.Series([e[0] for e in est]); io = pl.Series([e[1] for e in est])
    return (ctx.with_columns(lines_estimated=pl.col("spread").is_null() | (pl.col("week") > through_week),
                             _it=it, _io=io)
               .with_columns(
                   implied_team=pl.when("lines_estimated").then(pl.col("_it")).otherwise(pl.col("implied_team")),
                   implied_opp=pl.when("lines_estimated").then(pl.col("_io")).otherwise(pl.col("implied_opp")))
               .with_columns(
                   spread=pl.when("lines_estimated").then(pl.col("implied_team") - pl.col("implied_opp")).otherwise(pl.col("spread")),
                   total_line=pl.when("lines_estimated").then(pl.col("implied_team") + pl.col("implied_opp")).otherwise(pl.col("total_line")))
               .drop("_it", "_io"))


# ---------------------------------------------------------------- proyección del resto de la temporada

def season_state(src: dict, rosters_weekly: pl.DataFrame, scoring_cfg: dict, season: int, week: int) -> dict:
    """Tablas de features con una fila de "estado actual" por jugador/equipo en la semana `week`.

    Incluye a quienes descansan esa semana y a los que están en IR o inactivos (para proyectarlos
    cuando vuelvan). Devuelve también la historia para entrenar y lo que hace falta para el contexto.
    """
    tg = F.team_games(src["schedules"])
    ctx = F.game_context(tg)
    ids = F.gsis_to_espn(src["playerids"])

    base = F.offense_base(src["player_stats"], src["snap_counts"], src["opportunity"], src["playerids"])
    up_off = F.upcoming_offense(rosters_weekly, tg, base, season, week, statuses=STATE_STATUSES, require_game=False)
    off = F.build_offense(pl.concat([base, up_off], how="diagonal_relaxed"), ctx)

    points_k = scoring.k_points(src["player_stats"], scoring_cfg).join(ids, on="player_id", how="left")
    up_k = F.upcoming_k(rosters_weekly, tg, points_k, season, week, statuses=STATE_STATUSES,
                        require_game=False).join(ids, on="player_id", how="left")
    k = F.build_k(pl.concat([points_k, up_k], how="diagonal_relaxed"), ctx)

    points_dst = scoring.dst_points(src["team_stats"], src["schedules"], scoring_cfg).join(F.dst_espn_ids(), on="team", how="left")
    up_dst = F.upcoming_dst(tg, points_dst, season, week, require_game=False).join(F.dst_espn_ids(), on="team", how="left")
    t_off = F.team_offense(tg, src["team_stats"], upcoming=(season, week))
    dst = F.build_dst(pl.concat([points_dst, up_dst], how="diagonal_relaxed"), t_off, ctx)

    nfl_status = (rosters_weekly.filter(pl.col("season") == season, pl.col("week") == week)
                  .select(player_id="gsis_id", nfl_status="status").drop_nulls("player_id").unique("player_id"))
    return {"frames": {"offense": off, "k": k, "dst": dst}, "base": base, "points_k": points_k,
            "team_offense": t_off, "team_games": tg, "context": ctx, "ids": ids, "nfl_status": nfl_status}


def project_rest_of_season(state: dict, params: dict, cal: LeagueCalendar, cfg: dict) -> pl.DataFrame:
    """Proyección (puntos si juega) de cada jugador/equipo para cada semana restante. Bye = 0.

    Devuelve una fila por (espn_id, semana) con la proyección, el rival y si las líneas son estimadas.
    """
    season, now, weeks = cal.season, cal.current_week, cal.weeks
    tg = state["team_games"]
    fut = future_context(state["context"], tg, season, weeks, through_week=now, cfg=cfg["lines"])
    ctx_cols = [c for c in F.CONTEXT_FEATS]
    schedule = pl.DataFrame({"week": weeks}, schema={"week": pl.Int32})

    allowed_now = (F.latest_rolling(F.ppr_allowed(state["base"]), ["defense", "position"], ["opp_ppr_allowed"], season=season)
                     .rename({"defense": "opponent_team"}))
    offense_now = F.latest_rolling(state["team_offense"], "team", ["points_scored", "sacks_allowed", "giveaways"], season=season)
    offense_now = offense_now.select(pl.col("team").alias("opponent"),
                                     *[pl.col(c).alias(f"opp_{c}") for c in offense_now.columns if c != "team"])

    out = []
    for g, df in state["frames"].items():
        train = df.filter(pl.col("y").is_not_null())
        mdl = M.fit(train, g, params[g])
        now_rows = df.filter(pl.col("season") == season, pl.col("week") == now)
        opp_col = "opponent" if g == "dst" else "opponent_team"
        # features dependientes del partido: se sustituyen semana a semana
        drop = ctx_cols + [opp_col] + [c for c in now_rows.columns if c.startswith("opp_")]
        rows = (now_rows.drop("week", *drop, strict=False).join(schedule, how="cross")
                  .join(fut.select("week", "team", "lines_estimated", *ctx_cols, opponent="opponent"), on=["week", "team"], how="left")
                  .with_columns(bye=pl.col("opponent").is_null(),
                                weeks_since_last=pl.when(pl.col("week") > now).then(pl.lit(1, pl.Int64))
                                                   .otherwise(pl.col("weeks_since_last"))))
        if g == "offense":
            rows = (rows.rename({"opponent": "opponent_team"})
                        .join(allowed_now, on=["opponent_team", "position"], how="left"))
        elif g == "k":
            rows = rows.rename({"opponent": "opponent_team"})
        else:
            rows = rows.join(offense_now, on="opponent", how="left")
        rows = rows.select(*[c for c in df.columns if c in rows.columns], "bye", "lines_estimated")
        pred = M.predict(mdl, rows.drop("bye", "lines_estimated"), g)  # columnas auxiliares, no son features
        entity = rows["team"] if g == "dst" else rows["player_id"]
        out.append(M.with_position(rows, g).select(
            "season", "week", group=pl.lit(g), entity_id=entity,
            espn_id=pl.col("espn_id") if "espn_id" in rows.columns else pl.lit(None, dtype=pl.Int64),
            name=pl.concat_str("team", pl.lit(" D/ST")) if g == "dst" else pl.col("player_display_name"),
            position="position", nfl_team="team",
            opponent=pl.col("opponent") if g == "dst" else pl.col("opponent_team"),
            bye="bye", lines_estimated="lines_estimated",
            xfp_l5=pl.col("xfp_l5") if "xfp_l5" in rows.columns else pl.lit(None, dtype=pl.Float64),
            proj=pl.when(pl.col("bye")).then(0.0).otherwise(pl.Series(pred))))

    proj = pl.concat(out, how="diagonal_relaxed")
    proj = (proj.join(state["ids"].rename({"player_id": "entity_id", "espn_id": "_eid"}), on="entity_id", how="left")
                .with_columns(espn_id=pl.coalesce("espn_id", "_eid")).drop("_eid")
                .join(state["nfl_status"].rename({"player_id": "entity_id"}), on="entity_id", how="left"))
    return proj


def with_expected_points(proj: pl.DataFrame, rosters: pl.DataFrame, cal: LeagueCalendar, cfg: dict,
                         T: np.ndarray) -> pl.DataFrame:
    """Cruza la proyección (que ya trae `play_rate` por jugador) con un roster y añade puntos esperados.

    `rosters` puede ser el de la liga (league_rosters) o el de agentes libres (league_free_agents).
    - p_play: probabilidad de jugar esa semana, marginal de la cadena de disponibilidad del jugador
      (availability_paths; `T` = absence_transitions). 0 en bye.
    - exp_points: proyección del modelo x p_play (lo que se usa para valorar).
    - espn_points: media por partido de ESPN si juega esa semana (lo que ve el otro manager).
      Solo se ajusta por el estado de lesión conocido (OUT/doubtful esta semana, IR), no por el historial.
    """
    status_p = cfg["status_play_prob"]
    wk, now = pl.col("week"), cal.current_week
    df = rosters.join(proj.drop("position", "name"), on="espn_id", how="left").with_columns(bye=pl.col("bye").fill_null(True))
    people = df.filter(pl.col("proj").is_not_null()).unique("espn_id").select(
        "espn_id", "position", "play_rate", "injury_status", "lineup_slot", "nfl_status")
    df = df.join(availability_paths(people, cal, cfg, T), on=["espn_id", "week"], how="left")
    on_ir = pl.col("injury_status").is_in(list(IR_STATUSES)) | (pl.col("lineup_slot") == "IR") | (pl.col("nfl_status") == "RES")
    known_out = (pl.when(pl.col("bye")).then(0.0)
                   .when(on_ir & (wk < now + cfg["ir_weeks"])).then(0.0)
                   .when(wk == now).then(pl.col("injury_status").replace_strict(status_p, default=1.0, return_dtype=pl.Float64))
                   .otherwise(1.0))
    return (df.with_columns(p_play=pl.when(pl.col("bye")).then(0.0).otherwise(pl.col("p_avail").fill_null(0.0)), p_espn=known_out)
              .drop("p_avail")
              .with_columns(exp_points=pl.col("proj") * pl.col("p_play"),
                            espn_points=pl.col("espn_avg_proj") * pl.col("p_espn")))


# ---------------------------------------------------------------- valor de un roster y de un trade

def _week_lineup_points(wk: pl.DataFrame, cand: pl.DataFrame | None, slots: dict, score: str) -> float:
    """Puntos esperados de la alineación óptima de una semana, con nivel de reemplazo.

    1. Titulares: los mejores disponibles del roster; los slots vacíos (bye, OUT, IR) se rellenan con el
       mejor agente libre de esa posición (lineup.optimal_lineup con `replacements`).
    2. Un titular con probabilidad de jugar p < 1 aporta p·proyección + (1−p)·(mejor reemplazo): el
       mejor jugador de la banca o agente libre elegible para ese slot que no sea titular. Cada
       reemplazo se usa para un solo titular. Es la aproximación en valor esperado de "si no juega,
       entra el siguiente"; el Monte Carlo de la etapa (b) lo simula exactamente.
    """
    p_col = SCORE_PROB[score]
    opt = L.optimal_lineup(wk, slots, score, cand)
    pool = wk if cand is None else pl.concat([wk, cand.select(wk.columns)], how="vertical_relaxed")
    info = {r["espn_id"]: r for r in pool.to_dicts()}
    starters = set(opt["espn_id"].drop_nulls().to_list())
    backups = sorted((r for r in info.values() if r["available"] and r["espn_id"] not in starters and r[score] is not None),
                     key=lambda r: -r[score])
    total, used = 0.0, set()
    # primero los titulares con más riesgo: son los que más necesitan un buen reemplazo
    rows = sorted((r for r in opt.to_dicts() if r["espn_id"] is not None), key=lambda r: info[r["espn_id"]][p_col])
    for r in rows:
        me = info[r["espn_id"]]
        total += me[score]
        p = me[p_col]
        if p < 1:
            allowed = L.FLEX_SLOTS.get(r["slot"], {r["slot"]})
            fb = next((b for b in backups if b["espn_id"] not in used and b["position"] in allowed), None)
            if fb:
                used.add(fb["espn_id"])
                total += (1 - p) * fb[score]
    return total


def team_week_points(players: pl.DataFrame, slots: dict, score: str, repl: pl.DataFrame | None = None,
                     per_position: int = 3) -> pl.DataFrame:
    """Puntos esperados de la alineación óptima de un roster en cada semana (byes, lesiones, reemplazo).

    `repl`: agentes libres (proyección semanal); se usan los `per_position` mejores de cada posición.
    """
    rows = []
    avail = lambda df: df.with_columns(available=~pl.col("bye") & pl.col(score).is_not_null() & (pl.col(score) > 0))
    for (week,), wk in players.partition_by("week", as_dict=True).items():
        cand = None
        if repl is not None:
            cand = avail(repl.filter(pl.col("week") == week, ~pl.col("espn_id").is_in(wk["espn_id"].to_list())))
            cand = cand.filter("available").sort(score, descending=True).group_by("position").head(per_position)
        rows.append({"week": week, "points": _week_lineup_points(avail(wk), cand, slots, score)})
    return pl.DataFrame(rows, schema={"week": pl.Int32, "points": pl.Float64}).sort("week")


def weekly_lineups(players: pl.DataFrame, slots: dict, score: str, repl: pl.DataFrame | None = None,
                   per_position: int = 3) -> pl.DataFrame:
    """Alineación óptima de cada semana: slot, jugador, origen (roster/reemplazo) y puntos esperados."""
    avail = lambda df: df.with_columns(available=~pl.col("bye") & pl.col(score).is_not_null() & (pl.col(score) > 0))
    out = []
    for (week,), wk in players.partition_by("week", as_dict=True).items():
        cand = None
        if repl is not None:
            cand = avail(repl.filter(pl.col("week") == week, ~pl.col("espn_id").is_in(wk["espn_id"].to_list())))
            cand = cand.filter("available").sort(score, descending=True).group_by("position").head(per_position)
        opt = L.optimal_lineup(avail(wk), slots, score, cand).with_row_index("orden")
        pool = wk if cand is None else pl.concat([wk, cand.select(wk.columns)], how="vertical_relaxed")
        out.append(opt.join(pool.select("espn_id", "name", "position", score, "p_play"), on="espn_id", how="left")
                      .with_columns(week=pl.lit(week, pl.Int32)))
    return pl.concat(out).sort("week", "orden").drop("orden")


def lineup_changes_by_week(before: pl.DataFrame, after: pl.DataFrame, score: str = "exp_points") -> pl.DataFrame:
    """Qué titulares entran y salen cada semana (diferencia de conjuntos, sin importar el slot).

    `before`/`after`: salidas de weekly_lineups. Solo lista semanas con algún cambio.
    """
    rows = []
    for w in sorted(set(before["week"].to_list())):
        b = before.filter(pl.col("week") == w)
        a = after.filter(pl.col("week") == w)
        tag = lambda r: r["name"] + (" (agente libre)" if r["source"] == "reemplazo" else "")
        bs = {r["name"]: r for r in b.to_dicts() if r["name"]}
        as_ = {r["name"]: r for r in a.to_dicts() if r["name"]}
        out_, in_ = sorted(set(bs) - set(as_)), sorted(set(as_) - set(bs))
        if out_ or in_:
            rows.append({"week": w, "sale": ", ".join(tag(bs[n]) for n in out_), "entra": ", ".join(tag(as_[n]) for n in in_),
                         "puntos_esperados_antes": round(b[score].fill_null(0).sum(), 1),
                         "puntos_esperados_despues": round(a[score].fill_null(0).sum(), 1)})
    return pl.DataFrame(rows, schema={"week": pl.Int32, "sale": pl.Utf8, "entra": pl.Utf8,
                                      "puntos_esperados_antes": pl.Float64, "puntos_esperados_despues": pl.Float64})


def roster_value(players: pl.DataFrame, slots: dict, cal: LeagueCalendar, score: str, repl: pl.DataFrame | None = None) -> dict:
    wp = team_week_points(players, slots, score, repl).join(cal.weights(), on="week")
    return {"value": (wp["points"] * wp["weight"]).sum(),
            "regular": wp.filter(pl.col("phase") == "regular")["points"].sum(),
            "playoffs": wp.filter(pl.col("phase") == "playoffs")["points"].sum()}


def _apply_roster_limits(players: pl.DataFrame, incoming: set, rules: dict, cfg: dict, cal, score, repl=None):
    """Si sobran jugadores activos, suelta al que menos valor aporta (nunca a uno recién recibido).

    Devuelve (roster, jugador soltado o None, motivo de invalidez o None).
    """
    active = players.filter(pl.col("lineup_slot") != "IR").select("espn_id").unique().height
    dropped = None
    if active > cfg["max_active_roster"]:
        base_val = roster_value(players, rules["slots"], cal, score, repl)["value"]
        candidates = players.filter(~pl.col("espn_id").is_in(list(incoming)), pl.col("lineup_slot") != "IR")["espn_id"].unique()
        losses = {pid: base_val - roster_value(players.filter(pl.col("espn_id") != pid), rules["slots"], cal, score, repl)["value"]
                  for pid in candidates}
        drop_id = min(losses, key=losses.get)
        dropped = players.filter(pl.col("espn_id") == drop_id)["name"][0]
        players = players.filter(pl.col("espn_id") != drop_id)
    counts = players.select("espn_id", "position").unique().group_by("position").len()
    over = [f"{r['position']} ({r['len']} > {rules['position_limits'][r['position']]})" for r in counts.to_dicts()
            if r["position"] in rules["position_limits"] and r["len"] > rules["position_limits"][r["position"]]]
    return players, dropped, (f"excede el límite de {', '.join(over)}" if over else None)


def rosters_after_trade(league_proj: pl.DataFrame, me: int, other: int, gives: list[int], gets: list[int],
                        rules: dict, cal: LeagueCalendar, cfg: dict, score: str = "exp_points", repl=None):
    """Roster de `me` antes y después de dar `gives` y recibir `gets` (espn_id), aplicando límites.

    Los jugadores recibidos entran a la banca (slot BE), así que cuentan para el límite de activos.
    Devuelve (antes, después, jugador soltado o None, motivo de invalidez o None).
    """
    before = league_proj.filter(pl.col("fantasy_team_id") == me)
    received = (league_proj.filter(pl.col("fantasy_team_id") == other, pl.col("espn_id").is_in(gets))
                           .with_columns(fantasy_team_id=pl.lit(me), lineup_slot=pl.lit("BE")))
    after = pl.concat([before.filter(~pl.col("espn_id").is_in(gives)), received], how="vertical_relaxed")
    after, dropped, invalid = _apply_roster_limits(after, set(gets), rules, cfg, cal, score, repl)
    return before, after, dropped, invalid


def evaluate_trade(league_proj: pl.DataFrame, team_a: int, team_b: int, a_gives: list[int], b_gives: list[int],
                   rules: dict, cal: LeagueCalendar, cfg: dict, score: str = "exp_points", repl=None) -> pl.DataFrame:
    """Cambio en el valor del roster de cada equipo si A da `a_gives` y B da `b_gives` (espn_id).

    `repl`: proyección semanal de agentes libres para rellenar slots vacíos (nivel de reemplazo).
    """
    rows = []
    names = lambda ids, df: ", ".join(df.filter(pl.col("espn_id").is_in(ids))["name"].unique(maintain_order=True).to_list())
    for me, other, gives, gets in ((team_a, team_b, a_gives, b_gives), (team_b, team_a, b_gives, a_gives)):
        before, after, dropped, invalid = rosters_after_trade(league_proj, me, other, gives, gets, rules, cal, cfg, score, repl)
        v0, v1 = roster_value(before, rules["slots"], cal, score, repl), roster_value(after, rules["slots"], cal, score, repl)
        rows.append({"fantasy_team_id": me, "fantasy_team": before["fantasy_team"][0],
                     "da": names(gives, before), "recibe": names(gets, league_proj), "suelta": dropped,
                     "valor_antes": round(v0["value"], 1), "valor_despues": round(v1["value"], 1),
                     "delta": round(v1["value"] - v0["value"], 1),
                     "delta_regular": round(v1["regular"] - v0["regular"], 1),
                     "delta_playoffs": round(v1["playoffs"] - v0["playoffs"], 1),
                     "valido": invalid is None, "motivo": invalid})
    return pl.DataFrame(rows)


def weekly_trade_breakdown(league_proj: pl.DataFrame, me: int, other: int, gives: list[int], gets: list[int],
                           rules: dict, cal: LeagueCalendar, cfg: dict, score: str = "exp_points", repl=None) -> pl.DataFrame:
    """Puntos de la alineación óptima de `me` semana a semana, antes y después del trade."""
    before, after, _, _ = rosters_after_trade(league_proj, me, other, gives, gets, rules, cal, cfg, score, repl)
    b = team_week_points(before, rules["slots"], score, repl).rename({"points": "antes"})
    a = team_week_points(after, rules["slots"], score, repl).rename({"points": "despues"})
    return (b.join(a, on="week").join(cal.weights(), on="week")
             .with_columns(delta=pl.col("despues") - pl.col("antes"))
             .with_columns(delta_ponderado=pl.col("delta") * pl.col("weight")))


def resolve_players(league_proj: pl.DataFrame, team_id: int, names: list[str]) -> list[int]:
    """espn_id de jugadores de un equipo a partir de su nombre (exacto, sin distinguir mayúsculas)."""
    roster = league_proj.filter(pl.col("fantasy_team_id") == team_id).select("espn_id", "name").unique()
    ids = []
    for n in names:
        hit = roster.filter(pl.col("name").str.strip_chars().str.to_lowercase() == n.strip().lower())
        if hit.is_empty():
            raise ValueError(f"{n!r} no está en el roster del equipo {team_id}: {sorted(roster['name'].to_list())}")
        ids.append(hit["espn_id"][0])
    return ids


def resolve_team(league_proj: pl.DataFrame, name: str) -> int:
    teams = league_proj.select("fantasy_team_id", "fantasy_team").unique()
    hit = teams.filter(pl.col("fantasy_team").str.strip_chars().str.to_lowercase() == name.strip().lower())
    if hit.is_empty():
        raise ValueError(f"Equipo {name!r} no encontrado: {sorted(teams['fantasy_team'].to_list())}")
    return hit["fantasy_team_id"][0]


# ================================================================ etapa (b): incertidumbre
# ---------------------------------------------------------------- backtest de horizonte

def truncate_sources(src: dict, season: int, week: int) -> dict:
    """Los datos tal como se conocían antes de la semana `week` de `season` (para backtests).

    Estadísticas: solo partidos anteriores. Calendario: se conservan los partidos de la temporada (se
    conocen de antemano) pero sin marcador desde `week`; las temporadas posteriores se eliminan.
    """
    before = (pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week))
    cast = lambda df: df.with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)) \
        if df.schema["season"] != pl.Int32 or df.schema["week"] != pl.Int32 else df
    out = {k: v for k, v in src.items()}
    for k in ("player_stats", "team_stats", "snap_counts", "opportunity"):
        out[k] = cast(src[k]).filter(before)
    sched = cast(src["schedules"]).filter(pl.col("season") <= season)
    future = (pl.col("season") == season) & (pl.col("week") >= week)
    out["schedules"] = sched.with_columns([pl.when(future).then(None).otherwise(pl.col(c)).alias(c)
                                           for c in ("home_score", "away_score", "result", "total")])
    return out


def historical_calendar(season: int, week: int, playoff_weight: float) -> LeagueCalendar:
    """Calendario con el formato de la liga (1-14 regular, 15-17 playoffs) para una temporada pasada."""
    return LeagueCalendar(season=season, current_week=week, reg_weeks=list(range(1, 15)), playoff_weeks=[15, 16, 17],
                          trade_deadline=datetime(season, 12, 1, tzinfo=timezone.utc), playoff_weight=playoff_weight)


def horizon_backtest(src: dict, rosters_weekly: pl.DataFrame, scoring_cfg: dict, params: dict, cfg: dict,
                     season: int, origins=(3, 6, 9, 12)) -> pl.DataFrame:
    """Proyecta el resto de la temporada desde varias semanas de `season` y compara con lo que pasó.

    En cada origen W se usa solo lo que se sabía antes de W (truncate_sources) y el mismo método que en
    producción (features congeladas, líneas estimadas desde W+1). Devuelve una fila por jugador/equipo y
    semana jugada con la proyección, el resultado real y el horizonte (semanas desde W).
    """
    base_full = F.offense_base(src["player_stats"], src["snap_counts"], src["opportunity"], src["playerids"])
    actual = pl.concat([
        base_full.select(*KEYS, entity_id="player_id", y="fantasy_points_ppr"),
        scoring.k_points(src["player_stats"], scoring_cfg).select(*KEYS, entity_id="player_id", y="fantasy_points"),
        scoring.dst_points(src["team_stats"], src["schedules"], scoring_cfg).select(*KEYS, entity_id="team", y="fantasy_points"),
    ]).filter(pl.col("season") == season)
    out = []
    for w in origins:
        cut = truncate_sources(src, season, w)
        state = season_state(cut, rosters_weekly.filter(pl.col("season") == season), scoring_cfg, season, w)
        proj = project_rest_of_season(state, params, historical_calendar(season, w, cfg["playoff_weight"]), cfg)
        out.append(proj.filter(~pl.col("bye"))
                       .join(actual, on=["season", "week", "entity_id"], how="inner")
                       .with_columns(origin=pl.lit(w, pl.Int32), horizon=pl.col("week") - w,
                                     resid=pl.col("y") - pl.col("proj")))
    return pl.concat(out, how="diagonal_relaxed")


# ---------------------------------------------------------------- parámetros de incertidumbre

HORIZON_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 9), (10, 99)]
N_PROJ_BUCKETS = 5


def _horizon_bucket(k: int) -> int:
    return next(i for i, (lo, hi) in enumerate(HORIZON_BUCKETS) if lo <= k <= hi)


def persistent_share(horizon_bt: pl.DataFrame, min_weeks: int = 4) -> float:
    """Fracción del error que es persistente por jugador (ANOVA de efectos aleatorios sobre los residuos).

    Se agrupa por (origen, jugador): entre = varianza de las medias − ruido; dentro = varianza media.
    """
    g = (horizon_bt.group_by("origin", "entity_id")
                   .agg(pl.col("resid").mean().alias("m"), pl.col("resid").var().alias("v"), pl.len().alias("n"))
                   .filter(pl.col("n") >= min_weeks))
    within = g["v"].mean()
    between = max(g["m"].var() - (g["v"] / g["n"]).mean(), 0.0)
    return float(between / (between + within))


def uncertainty_params(horizon_bt: pl.DataFrame, transitions: np.ndarray) -> dict:
    """Todo lo que necesita el Monte Carlo: errores del backtest de horizonte y la cadena de disponibilidad.

    - residuos reales (y − proyección) por posición y quintil de proyección, para muestrear el error
      conservando su asimetría (partidos explosivos) y que crezca con la proyección;
    - rho: fracción persistente del error por posición;
    - factor de horizonte: desviación del error a k semanas / desviación global;
    - matriz de transición de disponibilidad (jugando / inactivo / reserva); piso de puntos por posición.
    """
    hb = horizon_bt.with_columns(hbucket=pl.col("horizon").map_elements(_horizon_bucket, return_dtype=pl.Int64))
    sd_all = hb["resid"].std()
    hfac = {int(b): float(s / sd_all) for b, s in hb.group_by("hbucket").agg(pl.col("resid").std()).rows()}
    pools = {}
    for (pos,), d in hb.partition_by("position", as_dict=True).items():
        edges = np.quantile(d["proj"].to_numpy(), np.linspace(0, 1, N_PROJ_BUCKETS + 1)[1:-1])
        b = np.searchsorted(edges, d["proj"].to_numpy())
        res = d["resid"].to_numpy()
        pools[pos] = {"edges": edges, "resid": [res[b == i] for i in range(N_PROJ_BUCKETS)],
                      "rho": persistent_share(d), "floor": float(d["y"].min())}
    return {"pools": pools, "horizon_factor": hfac, "transitions": transitions}


# ---------------------------------------------------------------- simulación

@dataclass
class Simulation:
    ids: np.ndarray          # espn_id de cada jugador simulado (P,)
    weeks: list[int]         # semanas simuladas (W)
    proj: np.ndarray         # proyección si juega (P, W); 0 en bye
    avail: np.ndarray        # juega esa semana (S, P, W)
    points: np.ndarray       # puntos si juega (S, P, W)

    def index(self, espn_ids) -> np.ndarray:
        pos = {e: i for i, e in enumerate(self.ids)}
        return np.array([pos[e] for e in espn_ids if e in pos], dtype=int)


def simulate(players: pl.DataFrame, cal: LeagueCalendar, cfg: dict, unc: dict, n_sims: int = 2000, seed: int = 0) -> Simulation:
    """Simula disponibilidad y puntos de cada jugador en cada semana restante.

    `players`: filas jugador-semana (liga + agentes libres) con espn_id, position, week, proj, bye,
    play_rate, injury_status, lineup_slot, nfl_status.
    Disponibilidad: la misma cadena de 3 estados (jugando, inactivo, reserva/IR) que availability_paths,
    así que la disponibilidad media simulada coincide con p_play del cálculo determinista. Puntos: proyección + error muestreado de los residuos del backtest de horizonte
    (por posición y quintil de proyección), con una parte persistente por jugador (rho) y escalado por
    el horizonte. D/ST siempre juega.
    """
    rng = np.random.default_rng(seed)
    weeks, now = cal.weeks, cal.current_week
    p = players.filter(pl.col("proj").is_not_null()).unique(["espn_id", "week"])
    ids = np.array(sorted(p["espn_id"].unique().to_list()))
    P, W, S = len(ids), len(weeks), n_sims
    grid = (pl.DataFrame({"espn_id": np.repeat(ids, W), "week": np.tile(np.array(weeks, dtype=np.int32), P)})
              .join(p, on=["espn_id", "week"], how="left")
              .with_columns(pl.col("bye").fill_null(True), pl.col("proj").fill_null(0.0)))
    proj = grid["proj"].to_numpy().reshape(P, W)
    bye = grid["bye"].to_numpy().reshape(P, W)
    info = p.unique("espn_id").sort("espn_id")
    pos = info["position"].to_list()
    rate = info["play_rate"].fill_null(1.0).to_numpy()
    on_ir = (info["injury_status"].is_in(list(IR_STATUSES)) | (info["lineup_slot"] == "IR") | (info["nfl_status"] == "RES")).fill_null(False).to_numpy()

    # --- disponibilidad: misma cadena de 3 estados que availability_paths (P, INA, RES)
    avail = np.zeros((S, P, W), dtype=bool)
    status = info["injury_status"].to_list()
    for i in range(P):
        if pos[i] == "D/ST":
            avail[:, i, :] = ~bye[i]
            continue
        Ti = player_chain(unc["transitions"], rate[i])
        cum = np.cumsum(Ti, axis=1)
        state = rng.choice(3, size=S, p=initial_state(status[i], bool(on_ir[i]), cfg))
        for j, w in enumerate(weeks):
            if j > 0:
                state = (rng.random(S)[:, None] > cum[state][:, :2]).sum(axis=1)   # muestreo de la fila de cada estado
            if on_ir[i] and w < now + cfg["ir_weeks"]:
                state[:] = 2
            avail[:, i, j] = (state == 0) & ~bye[i, j]

    # --- puntos si juega
    points = np.zeros((S, P, W), dtype=np.float32)
    for i in range(P):
        pool = unc["pools"].get(pos[i])
        if pool is None:
            continue
        z = rng.standard_normal(S)                          # componente persistente del jugador
        for j, w in enumerate(weeks):
            if bye[i, j]:
                continue
            bkt = int(np.searchsorted(pool["edges"], proj[i, j]))
            res = pool["resid"][bkt]
            sigma = res.std()
            f = unc["horizon_factor"].get(_horizon_bucket(w - now), 1.0)
            e = rng.choice(res, size=S) - res.mean()   # centrado: el Monte Carlo mide incertidumbre, no recalibra
            r = f * (np.sqrt(pool["rho"]) * sigma * z + np.sqrt(1 - pool["rho"]) * e)
            points[:, i, j] = np.maximum(proj[i, j] + r, pool["floor"])
    return Simulation(ids=ids, weeks=list(weeks), proj=proj, avail=avail, points=points)


def _sim_lineup_points(sim: Simulation, roster: np.ndarray, fa: np.ndarray, positions: dict, slots: dict, j: int) -> np.ndarray:
    """Puntos reales de la alineación en la semana j de cada simulación (S,).

    Cada simulación elige titulares entre los que juegan esa semana, por PROYECCIÓN (no por resultado:
    sin ver el futuro); los slots vacíos se llenan con agentes libres (que también pueden no jugar).
    """
    S = sim.avail.shape[0]
    order_r = roster[np.argsort(-sim.proj[roster, j])]
    order_f = fa[np.argsort(-sim.proj[fa, j])] if len(fa) else fa
    cand = np.concatenate([order_r, order_f])
    is_fa = np.concatenate([np.zeros(len(order_r), bool), np.ones(len(order_f), bool)])
    cpos = np.array([positions[c] for c in cand])
    av = sim.avail[:, cand, j]
    pts = sim.points[:, cand, j]
    used = np.zeros_like(av)
    filled_total = np.zeros(S, dtype=np.float64)
    slot_list = L.starting_slots(slots)
    empty = np.ones((S, len(slot_list)), dtype=bool)
    rows = np.arange(S)
    for fa_pass in (False, True):
        for k, slot in enumerate(slot_list):
            allowed = L.FLEX_SLOTS.get(slot, {slot})
            elig = np.isin(cpos, list(allowed)) & (is_fa == fa_pass)
            ok = av & ~used & elig[None, :] & empty[:, k:k + 1]
            has = ok.any(axis=1)
            idx = ok.argmax(axis=1)
            used[rows[has], idx[has]] = True
            empty[has, k] = False
            filled_total[has] += pts[rows[has], idx[has]]
    return filled_total


def sim_roster_weekly(sim: Simulation, roster_ids, fa_by_week: dict, positions: dict, slots: dict) -> np.ndarray:
    """Puntos reales de la alineación en cada simulación y semana (S, W), sin ponderar."""
    roster = sim.index(roster_ids)
    out = np.zeros((sim.avail.shape[0], len(sim.weeks)))
    for j, w in enumerate(sim.weeks):
        fa = sim.index([e for e in fa_by_week.get(w, []) if e not in set(roster_ids)])
        out[:, j] = _sim_lineup_points(sim, roster, fa, positions, slots, j)
    return out


def sim_roster_value(sim: Simulation, roster_ids, fa_by_week: dict, positions: dict, slots: dict, cal: LeagueCalendar) -> np.ndarray:
    """Valor ponderado (playoffs ×peso) del roster en cada simulación (S,)."""
    weights = np.array([cal.weight(w) for w in sim.weeks])
    return sim_roster_weekly(sim, roster_ids, fa_by_week, positions, slots) @ weights


def fa_candidates(fa_proj: pl.DataFrame, per_position: int = 3) -> dict:
    """Por semana: los `per_position` mejores agentes libres de cada posición (por puntos esperados)."""
    top = (fa_proj.filter(~pl.col("bye"), pl.col("exp_points") > 0)
                  .sort("exp_points", descending=True).group_by("week", "position").head(per_position))
    return {w: g["espn_id"].to_list() for (w,), g in top.partition_by("week", as_dict=True).items()}


def summarize(x: np.ndarray) -> dict:
    """Media (con su error de Monte Carlo), percentiles 10/50/90 y probabilidad de que sea positivo."""
    return {"media": float(x.mean()), "error_mc": float(x.std() / np.sqrt(len(x))), "p10": float(np.percentile(x, 10)),
            "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)), "prob_gana": float((x > 0).mean())}


def mc_evaluate_trade(sim: Simulation, league_proj: pl.DataFrame, fa_proj: pl.DataFrame, team_a: int, team_b: int,
                      a_gives: list[int], b_gives: list[int], rules: dict, cal: LeagueCalendar, cfg: dict,
                      return_draws: bool = False):
    """Distribución del cambio de valor de cada equipo (mismos sorteos antes y después del trade).

    El jugador que se suelta si sobra roster es el mismo que en la evaluación determinista.
    Con return_draws=True devuelve también, por equipo, los sorteos del cambio total (S,) y semanal (S, W).
    """
    positions = dict(zip(*pl.concat([league_proj, fa_proj], how="diagonal_relaxed")
                           .unique("espn_id").select("espn_id", "position").to_dict(as_series=False).values()))
    positions = {int(np.where(sim.ids == e)[0][0]): p for e, p in positions.items() if e in set(sim.ids)}
    fa_by_week = fa_candidates(fa_proj)
    rows, draws = [], {}
    for me, other, gives, gets in ((team_a, team_b, a_gives, b_gives), (team_b, team_a, b_gives, a_gives)):
        before, after, dropped, invalid = rosters_after_trade(league_proj, me, other, gives, gets, rules, cal, cfg,
                                                              "exp_points", fa_proj)
        w0 = sim_roster_weekly(sim, before["espn_id"].unique().to_list(), fa_by_week, positions, rules["slots"])
        w1 = sim_roster_weekly(sim, after["espn_id"].unique().to_list(), fa_by_week, positions, rules["slots"])
        weights = np.array([cal.weight(w) for w in sim.weeks])
        v0, v1 = w0 @ weights, w1 @ weights
        d = v1 - v0
        draws[me] = {"total": d, "weekly": w1 - w0}
        rows.append({"fantasy_team_id": me, "fantasy_team": before["fantasy_team"][0].strip(),
                     **{k: round(v, 3 if k == "prob_gana" else 1) for k, v in summarize(d).items()},
                     "valor_antes_p10": round(float(np.percentile(v0, 10)), 0), "valor_antes_p90": round(float(np.percentile(v0, 90)), 0),
                     "valido": invalid is None})
    out = pl.DataFrame(rows)
    return (out, draws) if return_draws else out


def mc_weekly_summary(draws_weekly: np.ndarray, sim: Simulation, cal: LeagueCalendar) -> pl.DataFrame:
    """Cambio semanal de puntos de la alineación (después − antes): media, intervalo 80% y P(>0)."""
    return pl.DataFrame({
        "week": pl.Series(sim.weeks, dtype=pl.Int32),
        "fase": ["playoffs" if w in cal.playoff_weeks else "regular" for w in sim.weeks],
        "media": draws_weekly.mean(axis=0).round(1), "p10": np.percentile(draws_weekly, 10, axis=0).round(1),
        "p90": np.percentile(draws_weekly, 90, axis=0).round(1), "prob_mejora": (draws_weekly > 0).mean(axis=0).round(2),
        "prob_empeora": (draws_weekly < 0).mean(axis=0).round(2)})
