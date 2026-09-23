"""Alineación óptima según los slots de la liga y comparación con la alineación actual."""
import polars as pl

# Slots flexibles: posiciones que aceptan
FLEX_SLOTS = {"RB/WR/TE": {"RB", "WR", "TE"}, "RB/WR": {"RB", "WR"}, "WR/TE": {"WR", "TE"},
              "OP": {"QB", "RB", "WR", "TE"}}
FIXED_SLOTS = ["QB", "RB", "WR", "TE", "K", "D/ST"]
BENCH_SLOTS = {"BE", "IR"}

# Estados de ESPN con los que el jugador no puede jugar
UNAVAILABLE = {"OUT", "INJURY_RESERVE", "SUSPENSION", "IR"}
DOUBTFUL = {"QUESTIONABLE", "DOUBTFUL"}


def starting_slots(slot_counts: dict) -> list[str]:
    """Lista de slots titulares (un elemento por plaza): primero los fijos, luego los flexibles."""
    fixed = [s for s in FIXED_SLOTS for _ in range(slot_counts.get(s, 0))]
    flex = [s for s in FLEX_SLOTS for _ in range(slot_counts.get(s, 0))]
    return fixed + flex


def optimal_lineup(players: pl.DataFrame, slot_counts: dict, score: str,
                   replacements: pl.DataFrame | None = None) -> pl.DataFrame:
    """Asigna los mejores jugadores disponibles a cada slot según `score`.

    Rellenar primero los slots fijos y después los flexibles es óptimo aquí, porque cada slot
    flexible acepta un superconjunto de las posiciones de los fijos.
    `players` necesita: espn_id, position, available (bool) y la columna `score`.

    `replacements` (opcional, mismas columnas): agentes libres que compiten por TODOS los slots con los
    jugadores del roster (nivel de reemplazo: un manager no alinea a alguien peor que el mejor agente
    libre disponible; lo ficharía). Así el valor de un roster nunca baja por tener un jugador más.
    Columna `source`: roster/reemplazo.
    """
    pool = players.filter(pl.col("available"), pl.col(score).is_not_null()).with_columns(source=pl.lit("roster"))
    if replacements is not None:
        pool = pl.concat([pool, replacements.filter(pl.col("available"), pl.col(score).is_not_null())
                                            .select(pool.drop("source").columns).with_columns(source=pl.lit("reemplazo"))],
                         how="vertical_relaxed")
    pool = pool.sort(score, descending=True).to_dicts()
    used, rows = set(), []
    for slot in starting_slots(slot_counts):
        allowed = FLEX_SLOTS.get(slot, {slot})
        pick = next((p for p in pool if p["espn_id"] not in used and p["position"] in allowed), None)
        if pick:
            used.add(pick["espn_id"])
        rows.append({"slot": slot, "espn_id": pick["espn_id"] if pick else None, "source": pick["source"] if pick else None})
    return pl.DataFrame(rows, schema={"slot": pl.Utf8, "espn_id": pl.Int64, "source": pl.Utf8})


def lineup_changes(roster: pl.DataFrame, optimal: pl.DataFrame, score: str, threshold: float) -> pl.DataFrame:
    """Cambios de titulares: quién entra y quién sale.

    Solo cuenta quién es titular, no en qué slot: mover a un RB del slot RB al FLEX no es un cambio.
    Cada jugador que entra se empareja con uno que sale de la misma posición si lo hay; si no,
    con el peor que sale (el cambio pasa por un slot flexible).
    """
    current = set(roster.filter(~pl.col("my_slot").is_in(list(BENCH_SLOTS)))["espn_id"].to_list())
    best = set(optimal["espn_id"].drop_nulls().to_list())
    info = {r["espn_id"]: r for r in roster.to_dicts()}
    ins = sorted((info[i] for i in best - current), key=lambda r: -r[score])
    outs = sorted((info[i] for i in current - best), key=lambda r: (r[score] is None, r[score] or 0))

    rows = []
    for p_in in ins:
        same = [o for o in outs if o["position"] == p_in["position"]]
        p_out = same[0] if same else (outs[0] if outs else None)
        if p_out:
            outs.remove(p_out)
        gain = p_in[score] - (p_out[score] or 0) if p_out else p_in[score]
        rows.append({"entra": p_in["name"], "pos_entra": p_in["position"],
                     "sale": p_out["name"] if p_out else "(slot vacío)", "pos_sale": p_out["position"] if p_out else None,
                     "motivo_salida": p_out["reason"] if p_out else "slot vacío",
                     "pred_entra": round(p_in[score], 1), "pred_sale": round(p_out[score], 1) if p_out and p_out[score] is not None else None,
                     "diferencia": round(gain, 1),
                     "concluyente": gain >= threshold or (p_out is not None and not p_out["available"])})
    return pl.DataFrame(rows, schema={"entra": pl.Utf8, "pos_entra": pl.Utf8, "sale": pl.Utf8, "pos_sale": pl.Utf8,
                                      "motivo_salida": pl.Utf8, "pred_entra": pl.Float64, "pred_sale": pl.Float64,
                                      "diferencia": pl.Float64, "concluyente": pl.Boolean})


def lineup_points(players: pl.DataFrame, slot_counts: dict, score: str,
                  replacements: pl.DataFrame | None = None) -> float:
    """Puntos totales (según `score`) de la alineación óptima (con reemplazos opcionales en slots vacíos)."""
    opt = optimal_lineup(players, slot_counts, score, replacements)
    pts = dict(zip(players["espn_id"].to_list(), players[score].to_list()))
    if replacements is not None:
        pts = {**dict(zip(replacements["espn_id"].to_list(), replacements[score].to_list())), **pts}
    return sum(pts[i] for i in opt["espn_id"].drop_nulls().to_list())


def pickup_gain(roster: pl.DataFrame, candidates: pl.DataFrame, slot_counts: dict, score: str) -> pl.Series:
    """Cuántos puntos sube mi alineación óptima si agrego cada candidato (0 si no entraría de titular)."""
    base = lineup_points(roster, slot_counts, score)
    cols = ["espn_id", "position", "available", score]
    gains = [lineup_points(pl.concat([roster.select(cols), cand.select(cols)]), slot_counts, score) - base
             for cand in candidates.iter_slices(1)]
    return pl.Series("mejora_alineacion", gains)
