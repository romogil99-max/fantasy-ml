"""Probabilidad de ganar el enfrentamiento de la semana contra mi rival (Monte Carlo).

- Proyecciones: las del registro de predicciones (última previa al partido): media y rango P10–P90.
- Alineaciones: la mía es la actual de ESPN (y la óptima como alternativa). La del rival es la actual,
  con sus huecos cubiertos: slots vacíos o titulares no disponibles (OUT, IR, suspendido, doubtful,
  bye o sin predicción) se llenan con su mejor suplente disponible (su alineación óptima en esos huecos).
- Simulación: cada jugador juega según su estado de ESPN; si juega, sus puntos se muestrean de los
  residuos reales del backtest de 2025 (por posición y quintil de predicción), escalados para que su
  dispersión coincida con SU rango P10–P90 y centrados en su predicción. Si un titular no juega, entra
  el mejor suplente disponible de su banca. Jugadores cuyo partido ya terminó: puntos reales de ESPN.
- Limitación: jugadores independientes entre sí (sin correlación QB–WR ni contra la D/ST rival).

Ejecutable: `python -m fantasy_ml.matchup --log` agrega una fila a data/predictions_log/winprob_<temporada>.csv.
"""
import argparse
from datetime import datetime, timezone

import numpy as np
import polars as pl

from . import data, espn, lineup as L, predictions_log as plog
from .data import DATA_PROC, PREDICTIONS_LOG

OUT_STATUSES = {"OUT", "INJURY_RESERVE", "IR", "SUSPENSION"}
BENCH = {"BE", "IR"}
N_BUCKETS = 5
THRESHOLD = 3.0  # decisiones "cerradas": menos de 3 puntos esperados de diferencia


# ---------------------------------------------------------------- datos del enfrentamiento

def espn_win_probability(league, week: int, me: int) -> float | None:
    """Probabilidad de ganar que muestra ESPN para mi equipo (None si no está disponible)."""
    try:
        raw = league.espn_request.league_get(params={"view": ["mMatchupScore", "mScoreboard"], "scoringPeriodId": week})
        m = next(x for x in raw["schedule"] if x.get("matchupPeriodId") == week
                 and me in (x.get("home", {}).get("teamId"), x.get("away", {}).get("teamId")))
        side = "home" if m["home"]["teamId"] == me else "away"
        return m[side].get("winProbability")
    except (StopIteration, KeyError, TypeError):
        return None


def matchup_players(league, week: int, me: int, log: pl.DataFrame, cfg: dict) -> tuple[pl.DataFrame, dict]:
    """Jugadores de los dos equipos con slot actual, estado, predicción del registro y puntos ya jugados."""
    box = next(b for b in league.box_scores(week)
               if me in (getattr(b.home_team, "team_id", None), getattr(b.away_team, "team_id", None)))
    home_is_me = box.home_team.team_id == me
    rival = box.away_team if home_is_me else box.home_team
    rows = []
    for team, lineup in ((box.home_team, box.home_lineup), (box.away_team, box.away_lineup)):
        for p in lineup:
            status = p.injuryStatus if isinstance(p.injuryStatus, str) else None
            rows.append({"team": "yo" if team.team_id == me else "rival", "espn_id": p.playerId, "name": p.name,
                         "position": p.position, "slot": p.slot_position, "injury": status,
                         "game_played": p.game_played, "espn_points": float(p.points or 0.0),
                         "espn_projection": float(p.projected_points or 0.0), "on_bye": bool(p.on_bye_week)})
    df = (pl.DataFrame(rows)
            .join(log.select("espn_id", "pred_model", "pred_q10", "pred_q90", "kickoff_utc"), on="espn_id", how="left"))
    status_p = cfg["status_play_prob"]
    df = df.with_columns(
        finished=pl.col("game_played") >= 100,
        p_play=pl.when(pl.col("on_bye") | pl.col("pred_model").is_null()).then(0.0)
                 .when(pl.col("injury").is_in(list(OUT_STATUSES))).then(0.0)
                 .otherwise(pl.col("injury").replace_strict(status_p, default=1.0, return_dtype=pl.Float64)))
    info = {"rival": rival.team_name.strip(), "espn_proj_me": box.away_projected if not home_is_me else box.home_projected,
            "espn_proj_rival": box.home_projected if not home_is_me else box.away_projected}
    return df, info


def fill_holes(team: pl.DataFrame, slots: dict) -> pl.DataFrame:
    """Alineación actual con los huecos cubiertos por el mejor suplente disponible (para el rival).

    Hueco: slot sin jugador, o titular con probabilidad de jugar < 0.5 (OUT, IR, suspendido, doubtful,
    bye, sin predicción). Devuelve el equipo con la columna `starter_slot` (None = banca).
    """
    starters = {r["espn_id"]: r["slot"] for r in team.filter(~pl.col("slot").is_in(list(BENCH))).to_dicts()}
    ok = {e for e, p in zip(team["espn_id"], team["p_play"]) if p >= 0.5}
    needed = L.starting_slots(slots)
    assigned, used = [], set()
    for slot in needed:  # titulares actuales disponibles, en su slot
        hit = next((e for e, s in starters.items() if s == slot and e in ok and e not in used), None)
        assigned.append(hit)
        if hit is not None:
            used.add(hit)
    bench = team.filter(~pl.col("espn_id").is_in(list(used)), pl.col("p_play") >= 0.5, pl.col("pred_model").is_not_null()) \
                .sort(pl.col("pred_model") * pl.col("p_play"), descending=True).to_dicts()
    for k, slot in enumerate(needed):
        if assigned[k] is None:
            pick = next((b for b in bench if b["espn_id"] not in used and b["position"] in L.FLEX_SLOTS.get(slot, {slot})), None)
            if pick:
                assigned[k] = pick["espn_id"]
                used.add(pick["espn_id"])
    slot_of = {e: s for e, s in zip(assigned, needed) if e is not None}
    return team.with_columns(starter_slot=pl.col("espn_id").replace_strict(slot_of, default=None, return_dtype=pl.Utf8))


def current_lineup(team: pl.DataFrame) -> pl.DataFrame:
    return team.with_columns(starter_slot=pl.when(pl.col("slot").is_in(list(BENCH))).then(None).otherwise(pl.col("slot")))


def optimal_lineup(team: pl.DataFrame, slots: dict) -> pl.DataFrame:
    """Alineación óptima por puntos esperados (predicción × probabilidad de jugar; ya jugados: puntos reales)."""
    t = team.with_columns(exp=pl.when(pl.col("finished")).then(pl.col("espn_points"))
                                 .otherwise(pl.col("pred_model").fill_null(0) * pl.col("p_play")),
                          available=(pl.col("p_play") > 0) | pl.col("finished"))
    opt = L.optimal_lineup(t, slots, "exp")
    slot_of = dict(zip(opt["espn_id"].to_list(), opt["slot"].to_list()))
    return team.with_columns(starter_slot=pl.col("espn_id").replace_strict(slot_of, default=None, return_dtype=pl.Utf8))


# ---------------------------------------------------------------- simulación

def residual_pools(backtest: pl.DataFrame) -> dict:
    """Residuos reales (y − predicción) del backtest semanal fuera de muestra, por posición y quintil de predicción."""
    pools = {}
    for (pos,), d in backtest.partition_by("position", as_dict=True).items():
        edges = np.quantile(d["pred"].to_numpy(), np.linspace(0, 1, N_BUCKETS + 1)[1:-1])
        b = np.searchsorted(edges, d["pred"].to_numpy())
        res = (d["y"] - d["pred"]).to_numpy()
        pools[pos] = {"edges": edges, "resid": [res[b == i] for i in range(N_BUCKETS)], "floor": float(d["y"].min())}
    return pools


def simulate_players(players: pl.DataFrame, pools: dict, n_sims: int = 10000, seed: int = 0) -> dict:
    """Puntos simulados (si juega) y disponibilidad de cada jugador: {espn_id: (puntos (S,), juega (S,))}.

    Los residuos de su posición y nivel se escalan para que el ancho P10–P90 simulado sea el de SU rango
    (q90 − q10) y se centran en su predicción (la media simulada es la predicción del modelo).
    Partidos ya terminados: puntos reales y juega = True.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for r in players.to_dicts():
        if r["finished"]:
            out[r["espn_id"]] = (np.full(n_sims, r["espn_points"]), np.ones(n_sims, bool))
            continue
        plays = rng.random(n_sims) < r["p_play"]
        pool = pools.get(r["position"])
        if r["pred_model"] is None or pool is None:
            out[r["espn_id"]] = (np.zeros(n_sims), np.zeros(n_sims, bool))
            continue
        res = pool["resid"][int(np.searchsorted(pool["edges"], r["pred_model"]))]
        width_pool = max(np.percentile(res, 90) - np.percentile(res, 10), 1e-6)
        width = (r["pred_q90"] - r["pred_q10"]) if r["pred_q90"] is not None else width_pool
        e = rng.choice(res, size=n_sims)
        e = (e - res.mean()) * (width / width_pool)
        out[r["espn_id"]] = (np.maximum(r["pred_model"] + e, pool["floor"]), plays)
    return out


def lineup_totals(team: pl.DataFrame, sims: dict, slots: dict, n_sims: int) -> np.ndarray:
    """Puntos del equipo en cada simulación con la alineación dada (columna starter_slot).

    Si un titular no juega, entra el mejor suplente disponible elegible (por predicción), como haría el
    manager antes del partido; los partidos ya terminados no se pueden cambiar.
    """
    starters = team.filter(pl.col("starter_slot").is_not_null()).to_dicts()
    bench = (team.filter(pl.col("starter_slot").is_null(), pl.col("slot") != "IR")
                 .sort(pl.col("pred_model").fill_null(-1), descending=True).to_dicts())
    total = np.zeros(n_sims)
    used = np.zeros((n_sims, len(bench)), bool)
    for s in starters:
        pts, plays = sims[s["espn_id"]]
        total += np.where(plays, pts, 0.0)
        if s["finished"]:
            continue
        missing = ~plays
        allowed = L.FLEX_SLOTS.get(s["starter_slot"], {s["starter_slot"]})
        for k, b in enumerate(bench):
            if b["position"] not in allowed or b["finished"]:
                continue
            bp, bplay = sims[b["espn_id"]]
            take = missing & bplay & ~used[:, k]
            total += np.where(take, bp, 0.0)
            used[:, k] |= take
            missing &= ~take
    return total


def win_probability(me_team: pl.DataFrame, rival_team: pl.DataFrame, sims: dict, slots: dict, n_sims: int) -> dict:
    a = lineup_totals(me_team, sims, slots, n_sims)
    b = lineup_totals(rival_team, sims, slots, n_sims)
    return {"p_win": float((a > b).mean() + 0.5 * (a == b).mean()), "exp_me": float(a.mean()), "exp_rival": float(b.mean()),
            "me_p10": float(np.percentile(a, 10)), "me_p90": float(np.percentile(a, 90)),
            "rival_p10": float(np.percentile(b, 10)), "rival_p90": float(np.percentile(b, 90)), "_me": a, "_rival": b}


def close_decisions(me_team: pl.DataFrame, rival_team: pl.DataFrame, sims: dict, slots: dict, n_sims: int,
                    threshold: float = THRESHOLD) -> pl.DataFrame:
    """Decisiones cerradas: para cada titular mío, suplentes elegibles a menos de `threshold` puntos esperados.

    - `si_favorito` / `si_no_favorito`: la regla general. Si soy favorito conviene el de MENOS varianza
      (asegura el resultado); si no, el de MÁS varianza (necesito un partido grande).
    - `conviene_hoy`: la opción con más P(ganar) esta semana (Monte Carlo, mismos sorteos para ambas).
      Combina media y varianza con mi situación real (favorito o no, y por cuánto).
    - `decide`: si la elección de hoy la marcó la media (el elegido tiene más puntos esperados) o la
      varianza (tiene menos puntos esperados pero la dispersión que conviene según la regla). Si la
      diferencia de P(ganar) es menor que 2 errores de Monte Carlo (diferencia pareada), no hay
      preferencia real: se mantiene el titular.
    `desv_*`: desviación estándar de los puntos simulados de cada jugador (incluye la probabilidad de no jugar).
    """
    base = win_probability(me_team, rival_team, sims, slots, n_sims)
    favorite = base["p_win"] >= 0.5
    rows = []
    starters = me_team.filter(pl.col("starter_slot").is_not_null(), ~pl.col("finished")).to_dicts()
    bench = me_team.filter(pl.col("starter_slot").is_null(), pl.col("slot") != "IR", ~pl.col("finished"),
                           pl.col("pred_model").is_not_null(), pl.col("p_play") > 0).to_dicts()
    sd = lambda e: float(np.std(np.where(sims[e][1], sims[e][0], 0.0)))
    exp = lambda r: r["pred_model"] * r["p_play"]
    for s in starters:
        allowed = L.FLEX_SLOTS.get(s["starter_slot"], {s["starter_slot"]})
        for b in bench:
            if b["position"] not in allowed or s["pred_model"] is None or abs(exp(b) - exp(s)) >= threshold:
                continue
            swapped = me_team.with_columns(starter_slot=pl.when(pl.col("espn_id") == b["espn_id"]).then(pl.lit(s["starter_slot"]))
                                                          .when(pl.col("espn_id") == s["espn_id"]).then(None)
                                                          .otherwise(pl.col("starter_slot")))
            alt = win_probability(swapped, rival_team, sims, slots, n_sims)
            # diferencia pareada (mismos sorteos) y su error de Monte Carlo
            win = lambda r: (r["_me"] > r["_rival"]) + 0.5 * (r["_me"] == r["_rival"])
            d = win(alt) - win(base)
            diff, se = float(d.mean()), float(d.std() / np.sqrt(n_sims))
            low, high = (s, b) if sd(s["espn_id"]) <= sd(b["espn_id"]) else (b, s)
            rule_pick = low if favorite else high
            if abs(diff) < 2 * se:
                pick, other, decide = s, b, "sin diferencia (ruido de la simulación): se mantiene el titular"
            else:
                pick, other = (b, s) if diff > 0 else (s, b)
                if exp(pick) >= exp(other):
                    decide = "la media"
                elif pick is rule_pick:
                    decide = "la varianza (" + ("favorito → menos" if favorite else "no favorito → más") + ")"
                else:
                    decide = "otros factores (probabilidad de jugar, forma de la distribución)"
            rows.append({"slot": s["starter_slot"], "titular": s["name"], "alternativa": b["name"],
                         "esperado_titular": round(exp(s), 1), "esperado_alternativa": round(exp(b), 1),
                         "desv_titular": round(sd(s["espn_id"]), 1), "desv_alternativa": round(sd(b["espn_id"]), 1),
                         "si_favorito": low["name"], "si_no_favorito": high["name"],
                         "p_ganar_titular": round(base["p_win"], 3), "p_ganar_alternativa": round(alt["p_win"], 3),
                         "error_mc_diferencia": round(se, 4), "conviene_hoy": pick["name"], "decide": decide})
    return pl.DataFrame(rows, schema={"slot": pl.Utf8, "titular": pl.Utf8, "alternativa": pl.Utf8,
                                      "esperado_titular": pl.Float64, "esperado_alternativa": pl.Float64,
                                      "desv_titular": pl.Float64, "desv_alternativa": pl.Float64,
                                      "si_favorito": pl.Utf8, "si_no_favorito": pl.Utf8,
                                      "p_ganar_titular": pl.Float64, "p_ganar_alternativa": pl.Float64,
                                      "error_mc_diferencia": pl.Float64, "conviene_hoy": pl.Utf8, "decide": pl.Utf8})


def range_consistency(players: pl.DataFrame, sims: dict) -> pl.DataFrame:
    """Compara, jugador por jugador, los percentiles 10/90 simulados (si juega) con su rango P10–P90."""
    rows = []
    for r in players.filter(~pl.col("finished"), pl.col("pred_q10").is_not_null()).to_dicts():
        pts = sims[r["espn_id"]][0]
        rows.append({"name": r["name"], "position": r["position"], "pred": r["pred_model"], "q10": r["pred_q10"], "q90": r["pred_q90"],
                     "sim_p10": float(np.percentile(pts, 10)), "sim_p90": float(np.percentile(pts, 90)), "sim_media": float(pts.mean())})
    return pl.DataFrame(rows).with_columns(
        ancho_rango=pl.col("q90") - pl.col("q10"), ancho_sim=pl.col("sim_p90") - pl.col("sim_p10"))


# ---------------------------------------------------------------- flujo completo

def analyze(season: int = 2026, n_sims: int = 10000, seed: int = 0) -> dict:
    """Todo el análisis del enfrentamiento de la semana actual."""
    cfg = data.load_config("trades")
    league = espn.connect(season)
    me, week = espn.my_team_id(), league.current_week
    slots = {k: v for k, v in league.settings.position_slot_counts.items() if v}
    log = plog.latest_pregame(plog.read(season)).filter(pl.col("week") == week)
    if log.is_empty():
        raise RuntimeError(f"No hay predicciones registradas para la semana {week}: ejecuta el notebook 04")
    players, info = matchup_players(league, week, me, log, cfg)
    bt = pl.read_parquet(DATA_PROC / "backtest_predictions.parquet").filter(pl.col("season") == 2025)
    sims = simulate_players(players, residual_pools(bt), n_sims, seed)
    mine, rival = players.filter(pl.col("team") == "yo"), players.filter(pl.col("team") == "rival")
    rival_l = fill_holes(rival, slots)
    actual = win_probability(current_lineup(mine), rival_l, sims, slots, n_sims)
    optimal = win_probability(optimal_lineup(mine, slots), rival_l, sims, slots, n_sims)
    return {"season": season, "week": week, "info": info, "players": players, "sims": sims, "slots": slots,
            "mine_current": current_lineup(mine), "mine_optimal": optimal_lineup(mine, slots), "rival": rival_l,
            "actual": actual, "optimal": optimal, "espn_p_win": espn_win_probability(league, week, me),
            "decisions": close_decisions(current_lineup(mine), rival_l, sims, slots, n_sims), "n_sims": n_sims}


WINPROB_SCHEMA = {
    "season": pl.Int32, "week": pl.Int32, "generated_at_utc": pl.Datetime("us", "UTC"), "code_version": pl.Utf8,
    "rival": pl.Utf8, "exp_me": pl.Float64, "exp_rival": pl.Float64, "p_win_current": pl.Float64,
    "p_win_optimal": pl.Float64, "p_win_espn": pl.Float64, "espn_proj_me": pl.Float64, "espn_proj_rival": pl.Float64,
    "games_finished": pl.Int32, "close_decisions": pl.Utf8,
}


def log_row(res: dict, now: datetime | None = None) -> pl.DataFrame:
    """Agrega la probabilidad de ganar a data/predictions_log/winprob_<temporada>.csv (solo se agregan filas)."""
    now = now or datetime.now(timezone.utc)
    dec = res["decisions"]
    row = pl.DataFrame([{
        "season": res["season"], "week": res["week"], "generated_at_utc": now, "code_version": plog.code_version(),
        "rival": res["info"]["rival"], "exp_me": res["actual"]["exp_me"], "exp_rival": res["actual"]["exp_rival"],
        "p_win_current": res["actual"]["p_win"], "p_win_optimal": res["optimal"]["p_win"], "p_win_espn": res["espn_p_win"],
        "espn_proj_me": res["info"]["espn_proj_me"], "espn_proj_rival": res["info"]["espn_proj_rival"],
        "games_finished": int(sum(t.filter(pl.col("starter_slot").is_not_null())["finished"].sum() for t in (res["mine_current"], res["rival"]))),
        "close_decisions": "; ".join(f"{r['slot']} {r['titular']} vs {r['alternativa']}: {r['conviene_hoy']}" for r in dec.to_dicts()) if dec.height else "",
    }]).select([pl.col(c).cast(t) for c, t in WINPROB_SCHEMA.items()])
    path = PREDICTIONS_LOG / f"winprob_{res['season']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as f:
        row.write_csv(f, include_header=new)
    return row


def main():
    ap = argparse.ArgumentParser(description="Probabilidad de ganar el enfrentamiento de esta semana")
    ap.add_argument("--log", action="store_true", help="agregar el resultado a data/predictions_log/winprob_<temporada>.csv")
    ap.add_argument("--sims", type=int, default=10000)
    args = ap.parse_args()
    res = analyze(n_sims=args.sims)
    a, o = res["actual"], res["optimal"]
    print(f"Semana {res['week']} vs {res['info']['rival']}: P(ganar) alineación actual {a['p_win']:.1%} · óptima {o['p_win']:.1%} · "
          f"ESPN {res['espn_p_win'] if res['espn_p_win'] is not None else 'n/d'} · puntos esperados {a['exp_me']:.1f} vs {a['exp_rival']:.1f}")
    if args.log:
        log_row(res)
        path = PREDICTIONS_LOG / f"winprob_{res['season']}.csv"
        print(f"✓ agregado a {path}")


if __name__ == "__main__":
    main()
