"""Análisis de varias semanas para mi equipo (notebook 05 y dashboard).

Reutiliza el contexto del analizador de trades (trades.build_trade_context): proyección semana a semana
con byes y probabilidad de jugar, agentes libres y Monte Carlo de temporadas.

1. weekly_outlook: mis puntos por semana con la alineación óptima (determinista y rango 80% simulado).
2. bye_calendar / replacement_needs: titular, banca, bye o fuera por jugador y semana; huecos reales
   (slots que mi roster no puede llenar por byes o lesiones) y el mejor agente libre para cada uno.
   En 1, 2 y 4 los agentes libres SOLO cubren slots vacíos (fill_only_empty): se describe mi roster.
3. multiweek_pickups: cuánto sube mi alineación en el horizonte si ficho a cada agente libre (soltando al
   que menos aporta si el roster está lleno). SIN nivel de reemplazo: la acción real de fichar.
4. upcoming_matchups: probabilidad de ganar contra cada rival de las próximas semanas (calendario de la
   liga), con los mismos sorteos del Monte Carlo para los dos equipos.
"""
import numpy as np
import polars as pl

from . import espn, trades as T

HORIZON = 4


def horizon_weeks(cal, horizon: int = HORIZON) -> list[int]:
    return cal.weeks[:horizon]


def _positions(sim, ctx) -> dict:
    both = pl.concat([ctx["league_proj"], ctx["fa_proj"]], how="diagonal_relaxed").unique("espn_id")
    pos = dict(both.select("espn_id", "position").iter_rows())
    return {i: pos[e] for i, e in enumerate(sim.ids) if e in pos}


def my_roster(ctx) -> pl.DataFrame:
    return ctx["league_proj"].filter(pl.col("fantasy_team_id") == ctx["me"])


# ---------------------------------------------------------------- 1. puntos por semana

def weekly_outlook(ctx, horizon: int = HORIZON) -> pl.DataFrame:
    """Puntos esperados de mi alineación óptima por semana (huecos cubiertos por agentes libres) y su rango 80% simulado."""
    weeks, slots = horizon_weeks(ctx["cal"], horizon), ctx["rules"]["slots"]
    mine = my_roster(ctx).filter(pl.col("week").is_in(weeks))
    det = T.team_week_points(mine, slots, "exp_points", T.prepare_replacements(ctx["fa_proj"]), fill_only_empty=True)
    sim = ctx["sim"]
    draws = T.sim_roster_weekly(sim, my_roster(ctx)["espn_id"].unique().to_list(), T.fa_candidates(ctx["fa_proj"]),
                                _positions(sim, ctx), slots, fill_only_empty=True)
    col = {w: j for j, w in enumerate(sim.weeks)}
    rows = [{"week": w, "esperado": float(det.filter(pl.col("week") == w)["points"][0]),
             "p10": float(np.percentile(draws[:, col[w]], 10)), "p90": float(np.percentile(draws[:, col[w]], 90))}
            for w in weeks]
    return pl.DataFrame(rows, schema_overrides={"week": pl.Int32})


# ---------------------------------------------------------------- 2. byes y huecos

def bye_calendar(ctx, last_week: int | None = None) -> pl.DataFrame:
    """Estado de cada jugador de mi roster por semana: T (titular en la óptima), B (banca), bye, fuera."""
    slots = ctx["rules"]["slots"]
    weeks = [w for w in ctx["cal"].weeks if last_week is None or w <= last_week]
    mine = my_roster(ctx).filter(pl.col("week").is_in(weeks))
    lu = T.weekly_lineups(mine, slots, "exp_points", T.prepare_replacements(ctx["fa_proj"]), fill_only_empty=True)
    starters = set(zip(lu.filter(pl.col("source") == "roster")["week"].to_list(), lu.filter(pl.col("source") == "roster")["espn_id"].to_list()))
    cell = [("bye" if r["bye"] else "fuera" if (r["p_play"] or 0) == 0 else "T" if (r["week"], r["espn_id"]) in starters else "B")
            for r in mine.select("week", "espn_id", "bye", "p_play").to_dicts()]
    wide = (mine.select("name", "position", "week").with_columns(estado=pl.Series(cell))
                .pivot(on="week", index=["name", "position"], values="estado", sort_columns=True))
    order = {p: i for i, p in enumerate(["QB", "RB", "WR", "TE", "K", "D/ST"])}
    return wide.with_columns(_o=pl.col("position").replace_strict(order, default=9)).sort("_o", "name").drop("_o")


def replacement_needs(ctx, horizon: int | None = None) -> pl.DataFrame:
    """Huecos reales: semanas y slots que mi roster no puede llenar (bye, lesión) y el mejor agente libre para cada uno."""
    slots = ctx["rules"]["slots"]
    weeks = ctx["cal"].weeks if horizon is None else horizon_weeks(ctx["cal"], horizon)
    mine = my_roster(ctx).filter(pl.col("week").is_in(weeks))
    lu = T.weekly_lineups(mine, slots, "exp_points", T.prepare_replacements(ctx["fa_proj"]), fill_only_empty=True)
    byes = (mine.filter("bye").group_by("week").agg(pl.col("name").sort().str.join(", ").alias("byes_del_roster")))
    return (lu.filter(pl.col("source") == "reemplazo")
              .select("week", "slot", agente_libre="name", pos="position", esperado=pl.col("exp_points").round(1))
              .join(byes, on="week", how="left").sort("week", "slot"))


# ---------------------------------------------------------------- 3. agentes libres a varias semanas

def multiweek_pickups(ctx, horizon: int = HORIZON, per_position: int = 6, max_active: int | None = None) -> pl.DataFrame:
    """Ganancia de fichar a cada agente libre en las próximas `horizon` semanas (suma de puntos esperados).

    Sin nivel de reemplazo (roster frente a roster + agente libre): es la acción real de fichar. Si el roster
    queda con más activos que el máximo, se suelta al jugador cuya salida más conviene (la mejor opción).
    `cubre_byes`: semanas del horizonte en que entra de titular mientras un titular mío descansa.
    """
    weeks, slots = horizon_weeks(ctx["cal"], horizon), ctx["rules"]["slots"]
    max_active = max_active or ctx["cfg"]["max_active_roster"]
    mine = my_roster(ctx).filter(pl.col("week").is_in(weeks))
    value = lambda df: float(T.team_week_points(df, slots, "exp_points")["points"].sum())
    base = value(mine)
    active = mine.filter(pl.col("lineup_slot") != "IR")["espn_id"].n_unique()
    droppable = mine.filter(pl.col("lineup_slot") != "IR")["espn_id"].unique().to_list()
    names = dict(mine.select("espn_id", "name").unique("espn_id").iter_rows())
    fa = ctx["fa_proj"].filter(pl.col("week").is_in(weeks), pl.col("position").is_in(["QB", "RB", "WR", "TE", "K", "D/ST"]))
    top = (fa.group_by("espn_id", "name", "position").agg(pl.col("exp_points").sum().alias("esperado_horizonte"))
             .sort("esperado_horizonte", descending=True).group_by("position").head(per_position))
    my_byes = mine.filter("bye").select("week", "position").unique()
    rows = []
    for r in top.to_dicts():
        add = fa.filter(pl.col("espn_id") == r["espn_id"]).with_columns(fantasy_team_id=pl.lit(ctx["me"]), lineup_slot=pl.lit("BE"))
        with_fa = pl.concat([mine, add.select(mine.columns)], how="vertical_relaxed")
        if active + 1 > max_active:
            options = {x: value(with_fa.filter(pl.col("espn_id") != x)) for x in droppable}
            drop = max(options, key=options.get)
            new_val, dropped = options[drop], names[drop]
        else:
            new_val, dropped = value(with_fa), None
        lu = T.weekly_lineups(with_fa if dropped is None else with_fa.filter(pl.col("espn_id") != drop), slots, "exp_points")
        starts = lu.filter(pl.col("espn_id") == r["espn_id"])["week"].to_list()
        covers = sorted(set(starts) & set(my_byes["week"].to_list()))
        rows.append({"agente_libre": r["name"], "pos": r["position"], "esperado_horizonte": round(r["esperado_horizonte"], 1),
                     "ganancia": round(new_val - base, 1), "suelto": dropped, "semanas_titular": starts, "cubre_byes": covers})
    return pl.DataFrame(rows).sort("ganancia", descending=True)


# ---------------------------------------------------------------- 4. próximos rivales

def upcoming_matchups(ctx, horizon: int = HORIZON) -> pl.DataFrame:
    """P(ganar) contra el rival de cada una de las próximas semanas de temporada regular.

    Ambos equipos con su alineación óptima semana a semana (elegida por proyección entre los que juegan;
    agentes libres solo para los huecos) y los mismos sorteos de disponibilidad y puntos para todos.
    """
    league = espn.connect(ctx["cal"].season)
    me_team = next(t for t in league.teams if t.team_id == ctx["me"])
    sim, slots = ctx["sim"], ctx["rules"]["slots"]
    fa_by_week, positions = T.fa_candidates(ctx["fa_proj"]), _positions(sim, ctx)
    col = {w: j for j, w in enumerate(sim.weeks)}
    roster_of = lambda tid: ctx["league_proj"].filter(pl.col("fantasy_team_id") == tid)["espn_id"].unique().to_list()
    mine = T.sim_roster_weekly(sim, roster_of(ctx["me"]), fa_by_week, positions, slots, fill_only_empty=True)
    cache, rows = {}, []
    for w in horizon_weeks(ctx["cal"], horizon):
        if w not in ctx["cal"].reg_weeks or w - 1 >= len(me_team.schedule):
            continue
        opp = me_team.schedule[w - 1]
        if opp.team_id not in cache:
            cache[opp.team_id] = T.sim_roster_weekly(sim, roster_of(opp.team_id), fa_by_week, positions, slots, fill_only_empty=True)
        a, b = mine[:, col[w]], cache[opp.team_id][:, col[w]]
        rows.append({"week": w, "rival": opp.team_name.strip(), "yo_esperado": round(float(a.mean()), 1),
                     "rival_esperado": round(float(b.mean()), 1),
                     "p_ganar": round(float((a > b).mean() + 0.5 * (a == b).mean()), 3)})
    return pl.DataFrame(rows, schema_overrides={"week": pl.Int32})


def analyze(ctx, horizon: int = HORIZON) -> dict:
    return {"weekly": weekly_outlook(ctx, horizon), "calendar": bye_calendar(ctx), "needs": replacement_needs(ctx),
            "pickups": multiweek_pickups(ctx, horizon), "matchups": upcoming_matchups(ctx, horizon),
            "horizon": horizon_weeks(ctx["cal"], horizon)}
