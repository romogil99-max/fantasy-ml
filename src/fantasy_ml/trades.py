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


def play_probability(cal: LeagueCalendar, cfg: dict) -> pl.Expr:
    """Probabilidad de jugar cada semana: estado de ESPN esta semana, IR las próximas `ir_weeks`, tasa histórica después."""
    status_p = cfg["status_play_prob"]
    wk, now = pl.col("week"), cal.current_week
    on_ir = pl.col("injury_status").is_in(list(IR_STATUSES)) | (pl.col("lineup_slot") == "IR") | (pl.col("nfl_status") == "RES")
    this_week = pl.col("injury_status").replace_strict(status_p, default=1.0, return_dtype=pl.Float64)
    return (pl.when(pl.col("bye")).then(0.0)
              .when(on_ir & (wk < now + cfg["ir_weeks"])).then(0.0)
              .when(wk == now).then(this_week)
              .otherwise(pl.col("play_rate")))


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


def with_expected_points(proj: pl.DataFrame, rosters: pl.DataFrame, cal: LeagueCalendar, cfg: dict) -> pl.DataFrame:
    """Cruza la proyección (que ya trae `play_rate` por jugador) con un roster y añade puntos esperados.

    `rosters` puede ser el de la liga (league_rosters) o el de agentes libres (league_free_agents).

    - exp_points: proyección del modelo x probabilidad de jugar (lo que se usa para valorar).
    - espn_points: media por partido de ESPN si juega esa semana (lo que ve el otro manager).
      Solo se ajusta por el estado de lesión conocido (OUT esta semana, IR), no por la tasa histórica.
    """
    status_p = cfg["status_play_prob"]
    wk, now = pl.col("week"), cal.current_week
    df = rosters.join(proj.drop("position", "name"), on="espn_id", how="left").with_columns(bye=pl.col("bye").fill_null(True))
    on_ir = pl.col("injury_status").is_in(list(IR_STATUSES)) | (pl.col("lineup_slot") == "IR") | (pl.col("nfl_status") == "RES")
    known_out = (pl.when(pl.col("bye")).then(0.0)
                   .when(on_ir & (wk < now + cfg["ir_weeks"])).then(0.0)
                   .when(wk == now).then(pl.col("injury_status").replace_strict(status_p, default=1.0, return_dtype=pl.Float64))
                   .otherwise(1.0))
    return (df.with_columns(p_play=play_probability(cal, cfg), p_espn=known_out)
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
