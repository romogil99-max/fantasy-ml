"""Dashboard de fantasy-ml: probabilidad de ganar, predicciones y sugerencias de la semana.

Ejecutar:  streamlit run app/dashboard.py
Servicio:  systemd/fantasy-dashboard.service (puerto 8501, red local, sin autenticación).

Los datos de ESPN se consultan en vivo y se guardan en caché 10 minutos (botón "Actualizar" en la barra
lateral). Las predicciones salen del registro (data/predictions_log): las mismas que quedan fechadas en git.
"""
import json
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import altair as alt
import polars as pl
import streamlit as st

from fantasy_ml import espn, lineup as L, matchup as MU, outlook as O, predictions_log as plog, report, trades as T
from fantasy_ml.data import DATA_PROC, PREDICTIONS_LOG

GDL = ZoneInfo("America/Mexico_City")
SEASON = 2026
# Paleta de referencia (skill dataviz): slot 1 azul = modelo, slot 2 naranja = ESPN. Orden fijo.
C_MODEL, C_ESPN = "#2a78d6", "#eb6834"
FINDER_PATH = DATA_PROC / "trade_finder_latest.parquet"
FINDER_META = DATA_PROC / "trade_finder_latest.json"

st.set_page_config(page_title="fantasy-ml", page_icon="🏈", layout="wide")


# ---------------------------------------------------------------- datos (con caché)

@st.cache_resource(ttl=600, show_spinner="Consultando ESPN y simulando el enfrentamiento…")
def load_matchup():
    return MU.analyze(season=SEASON, n_sims=10000)


@st.cache_resource(ttl=600, show_spinner="Buscando agentes libres…")
def load_free_agents(week: int):
    league = espn.connect(SEASON)
    log = plog.latest_pregame(plog.read(SEASON)).filter(pl.col("week") == week)
    return (espn.free_agent_projections(league, week)
            .select("espn_id", name="espn_name", position="espn_position", injury="espn_injury_status")
            .join(log.select("espn_id", "pred_model", "pred_q10", "pred_q90", "espn_projection", "opponent"), on="espn_id")
            .filter(~pl.col("injury").fill_null("").is_in(list(L.UNAVAILABLE))))


@st.cache_resource(ttl=3600, show_spinner="Cruzando el registro con los resultados reales…")
def load_evaluation():
    return report.logged_vs_actual(SEASON)


@st.cache_resource(ttl=3600, show_spinner="Proyectando las próximas semanas (~20 s)…")
def load_outlook(horizon: int):
    return O.analyze(T.build_trade_context(SEASON), horizon=horizon)


def read_winprob_log() -> pl.DataFrame:
    path = PREDICTIONS_LOG / f"winprob_{SEASON}.csv"
    return pl.read_csv(path, schema_overrides=MU.WINPROB_SCHEMA) if path.exists() else pl.DataFrame(schema=MU.WINPROB_SCHEMA)


def pct(x):
    return "—" if x is None else f"{x:.0%}"


# ---------------------------------------------------------------- barra lateral

with st.sidebar:
    st.title("🏈 fantasy-ml")
    if st.button("🔄 Actualizar datos de ESPN", width="stretch"):
        st.cache_resource.clear()
        st.rerun()
    st.caption(f"Actualizado: {datetime.now(GDL):%d-%m %H:%M} (Guadalajara)\n\nLos datos se refrescan solos cada 10 min.")
    st.caption(f"Versión del código: `{plog.code_version()}`")

try:
    res = load_matchup()
except RuntimeError as e:  # sin predicciones registradas para la semana
    st.error(str(e))
    st.stop()

week, rival = res["week"], res["info"]["rival"]
a, o = res["actual"], res["optimal"]
players = res["players"]

tab_week, tab_lineup, tab_fa, tab_next, tab_trades, tab_model = st.tabs(
    ["📊 Esta semana", "📋 Mi alineación", "🆓 Agentes libres", "📅 Próximas semanas", "🔁 Trades", "🎯 ¿Qué tan bien va el modelo?"])

# ---------------------------------------------------------------- esta semana

with tab_week:
    st.header(f"Semana {week} vs {rival}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(ganar) · mi alineación", pct(a["p_win"]))
    c2.metric("P(ganar) · alineación óptima", pct(o["p_win"]),
              delta=f"{(o['p_win'] - a['p_win']) * 100:+.1f} pts" if abs(o["p_win"] - a["p_win"]) >= 0.005 else None)
    c3.metric("P(ganar) · ESPN", pct(res["espn_p_win"]))
    c4.metric("Puntos esperados", f"{a['exp_me']:.1f} vs {a['exp_rival']:.1f}",
              help=f"Rango 80%: yo {a['me_p10']:.0f}–{a['me_p90']:.0f} · rival {a['rival_p10']:.0f}–{a['rival_p90']:.0f}")
    st.caption(f"Monte Carlo de {res['n_sims']:,} simulaciones · ESPN proyecta {res['info']['espn_proj_me']:.1f} vs "
               f"{res['info']['espn_proj_rival']:.1f}. Los jugadores se simulan de forma independiente (sin correlación "
               "QB–receptor), así que las probabilidades son algo más extremas de lo real.")

    st.subheader("Probabilidad de ganar registrada")
    wp = read_winprob_log()
    if wp.is_empty():
        st.info("Aún no hay probabilidades registradas. La ejecución automática del domingo agrega una cada semana.")
    else:
        hist = (wp.select("week", "rival", "generated_at_utc", modelo="p_win_current", espn="p_win_espn")
                  .with_columns(momento=pl.col("generated_at_utc").dt.convert_time_zone("America/Mexico_City")
                                           .dt.strftime("%d-%m %H:%M"),
                                etiqueta=pl.format("S{} · {}", "week", pl.col("rival")))
                  .with_row_index("orden"))
        long = hist.unpivot(index=["orden", "week", "rival", "momento", "etiqueta"], on=["modelo", "espn"],
                            variable_name="fuente", value_name="p_ganar").with_columns(
            fuente=pl.col("fuente").replace({"modelo": "Modelo", "espn": "ESPN"}))
        if hist.height >= 2:
            base = alt.Chart(long.to_pandas()).encode(
                x=alt.X("orden:O", title=None, axis=alt.Axis(labelExpr="''", ticks=False)),
                y=alt.Y("p_ganar:Q", title="P(ganar)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", grid=True)),
                color=alt.Color("fuente:N", scale=alt.Scale(domain=["Modelo", "ESPN"], range=[C_MODEL, C_ESPN]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=[alt.Tooltip("etiqueta:N", title="Semana"), alt.Tooltip("momento:N", title="Registrado"),
                         alt.Tooltip("fuente:N", title="Fuente"), alt.Tooltip("p_ganar:Q", title="P(ganar)", format=".0%")])
            rule = alt.Chart().mark_rule(color="#8c8b87", strokeDash=[4, 4], strokeWidth=1).encode(y=alt.datum(0.5))
            chart = (base.mark_line(strokeWidth=2) + base.mark_point(size=80, filled=True, strokeWidth=2) + rule).properties(height=260)
            st.altair_chart(chart, width="stretch", theme="streamlit")
        st.dataframe(hist.select("etiqueta", "momento", "modelo", "espn").rename({"etiqueta": "semana", "momento": "registrado"}),
                     hide_index=True, width="stretch",
                     column_config={"modelo": st.column_config.NumberColumn("Modelo", format="percent"),
                                    "espn": st.column_config.NumberColumn("ESPN", format="percent")})

    st.subheader(f"Alineación de {rival}")
    st.caption("Su alineación actual en ESPN; los huecos (vacío, OUT, IR, doubtful, bye) se cubren con su mejor suplente.")
    st.dataframe(res["rival"].filter(pl.col("starter_slot").is_not_null())
                 .select(slot="starter_slot", jugador="name", pos="position", lesion="injury",
                         prediccion="pred_model", p10="pred_q10", p90="pred_q90", p_jugar="p_play", espn="espn_projection"),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("prediccion", "p10", "p90", "espn")}
                 | {"p_jugar": st.column_config.NumberColumn("P(jugar)", format="percent")})

# ---------------------------------------------------------------- mi alineación

with tab_lineup:
    st.header("Mi alineación")
    mine = res["mine_current"].join(res["mine_optimal"].select("espn_id", optima="starter_slot"), on="espn_id")
    order = {s: i for i, s in enumerate(["QB", "RB", "WR", "TE", "RB/WR/TE", "OP", "K", "D/ST", "BE", "IR"])}
    table = (mine.with_columns(_o=pl.col("slot").replace_strict(order, default=99))
                 .sort("_o", pl.col("pred_model"), descending=[False, True], nulls_last=True)
                 .select(slot="slot", jugador="name", pos="position", lesion="injury", prediccion="pred_model",
                         p10="pred_q10", p90="pred_q90", espn="espn_projection", p_jugar="p_play",
                         en_la_optima=pl.col("optima").is_not_null()))
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("prediccion", "p10", "p90", "espn")}
                 | {"p_jugar": st.column_config.NumberColumn("P(jugar)", format="percent"),
                    "en_la_optima": st.column_config.CheckboxColumn("¿Titular en la óptima?"),
                    "p10": st.column_config.NumberColumn("P10", format="%.1f", help="8 de cada 10 veces sus puntos caen entre P10 y P90"),
                    "p90": st.column_config.NumberColumn("P90", format="%.1f")})

    cur = set(res["mine_current"].filter(pl.col("starter_slot").is_not_null())["name"])
    opt = set(res["mine_optimal"].filter(pl.col("starter_slot").is_not_null())["name"])
    if cur == opt:
        st.success("✓ Tu alineación actual ya es la óptima según el modelo.")
    else:
        st.warning(f"**Cambios para la óptima** ({pct(a['p_win'])} → {pct(o['p_win'])}): "
                   f"entran {', '.join(sorted(opt - cur))} · salen {', '.join(sorted(cur - opt))}")

    st.subheader("Decisiones cerradas (< 3 puntos esperados)")
    st.caption("Si eres favorito conviene el de **menos varianza**; si no, el de **más**. "
               "*Conviene hoy* es la opción con más probabilidad de ganar esta semana (mismos sorteos); "
               "si la diferencia es menor que el ruido de la simulación, se mantiene el titular.")
    dec = res["decisions"]
    if dec.is_empty():
        st.info("No hay decisiones cerradas esta semana.")
    else:
        st.dataframe(dec.select("slot", "titular", "alternativa", "esperado_titular", "esperado_alternativa",
                                "si_favorito", "si_no_favorito", "p_ganar_titular", "p_ganar_alternativa", "conviene_hoy", "decide"),
                     hide_index=True, width="stretch",
                     column_config={"p_ganar_titular": st.column_config.NumberColumn("P(ganar) titular", format="percent"),
                                    "p_ganar_alternativa": st.column_config.NumberColumn("P(ganar) alternativa", format="percent")})

# ---------------------------------------------------------------- agentes libres

with tab_fa:
    st.header("Mejores agentes libres por posición")
    st.caption("Top 5 por predicción entre los agentes libres de ESPN con predicción en el registro. "
               "**Mejora de la alineación**: cuántos puntos sube mi alineación óptima si lo agrego (0 si no entraría de titular).")
    fa = load_free_agents(week)
    roster = res["mine_current"].with_columns(available=pl.col("p_play") > 0,
                                              pred_model=pl.col("pred_model").fill_null(0) * pl.col("p_play"))
    top = (fa.with_columns(available=pl.lit(True)).sort("pred_model", descending=True)
             .group_by("position", maintain_order=True).head(5))
    slots = res["slots"]
    top = top.with_columns(L.pickup_gain(roster.select("espn_id", "position", "available", "pred_model"), top, slots, "pred_model"))
    order = {s: i for i, s in enumerate(["QB", "RB", "WR", "TE", "K", "D/ST"])}
    top = top.with_columns(_o=pl.col("position").replace_strict(order, default=9)).sort("_o", pl.col("pred_model"), descending=[False, True])
    worth = top.filter(pl.col("mejora_alineacion") >= 3)
    if worth.height:
        st.success("**Vale la pena considerar:** " + ", ".join(f"{r['name']} ({r['position']}, +{r['mejora_alineacion']:.1f})"
                                                           for r in worth.to_dicts()))
    st.dataframe(top.select(pos="position", jugador="name", rival="opponent", lesion="injury", prediccion="pred_model",
                            p10="pred_q10", p90="pred_q90", espn="espn_projection", mejora="mejora_alineacion"),
                 hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("prediccion", "p10", "p90", "espn")}
                 | {"mejora": st.column_config.NumberColumn("Mejora de la alineación", format="%+.1f")})

# ---------------------------------------------------------------- próximas semanas

with tab_next:
    st.header("Próximas semanas")
    horizon = st.segmented_control("Horizonte", options=[2, 4, 6], default=4, format_func=lambda n: f"{n} semanas")
    out = load_outlook(horizon or 4)
    st.caption("Tu roster actual semana a semana: los agentes libres **solo cubren los huecos** (bye o lesión sin nadie "
               "disponible en tu roster). Proyección con features actuales y líneas de apuestas estimadas para semanas futuras.")

    left, right = st.columns(2)
    with left:
        st.subheader("Mis puntos por semana")
        wk = out["weekly"].with_columns(pl.col("esperado", "p10", "p90").round(1))
        base = alt.Chart(wk.to_pandas()).encode(x=alt.X("week:O", title="Semana", axis=alt.Axis(labelAngle=0)))
        tip = [alt.Tooltip("week:O", title="Semana"), alt.Tooltip("esperado:Q", title="Esperado", format=".1f"),
               alt.Tooltip("p10:Q", title="P10", format=".1f"), alt.Tooltip("p90:Q", title="P90", format=".1f")]
        chart = (base.mark_rule(color=C_MODEL, strokeWidth=4, opacity=0.35, strokeCap="round")
                     .encode(y=alt.Y("p10:Q", title="Puntos", scale=alt.Scale(zero=False)), y2="p90:Q", tooltip=tip)
                 + base.mark_point(color=C_MODEL, filled=True, size=90).encode(y="esperado:Q", tooltip=tip)
                 ).properties(height=260)
        st.altair_chart(chart, width="stretch", theme="streamlit")
        st.caption("Punto: puntos esperados · barra: rango del 80% (Monte Carlo).")
        st.dataframe(wk, hide_index=True, width="stretch",
                     column_config={"week": "Semana", "esperado": st.column_config.NumberColumn("Esperado", format="%.1f"),
                                    "p10": st.column_config.NumberColumn("P10", format="%.1f"),
                                    "p90": st.column_config.NumberColumn("P90", format="%.1f")})
    with right:
        st.subheader("Probabilidad de ganar contra mis próximos rivales")
        mt = out["matchups"]
        if mt.is_empty():
            st.info("No quedan semanas de temporada regular en el horizonte.")
        else:
            mtp = mt.with_columns(etiqueta=pl.format("S{} · {}", "week", "rival")).to_pandas()
            tip = [alt.Tooltip("etiqueta:N", title="Semana"), alt.Tooltip("p_ganar:Q", title="P(ganar)", format=".0%"),
                   alt.Tooltip("yo_esperado:Q", title="Yo (esperado)", format=".1f"),
                   alt.Tooltip("rival_esperado:Q", title="Rival (esperado)", format=".1f")]
            bars = (alt.Chart(mtp).mark_bar(color=C_MODEL, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=36)
                    .encode(x=alt.X("etiqueta:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, labelLimit=140)),
                            y=alt.Y("p_ganar:Q", title="P(ganar)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
                            tooltip=tip))
            half = alt.Chart().mark_rule(color="#8c8b87", strokeDash=[4, 4], strokeWidth=1).encode(y=alt.datum(0.5))
            st.altair_chart((bars + half).properties(height=260), width="stretch", theme="streamlit")
            st.dataframe(mt, hide_index=True, width="stretch",
                         column_config={"week": "Semana", "p_ganar": st.column_config.NumberColumn("P(ganar)", format="percent"),
                                        "yo_esperado": st.column_config.NumberColumn("Yo", format="%.1f"),
                                        "rival_esperado": st.column_config.NumberColumn("Rival", format="%.1f")})
            st.caption("Alineación óptima de los dos equipos. Para esta semana es más preciso el cálculo de la pestaña "
                       "“Esta semana”, que usa tu alineación real y el rango de cada jugador.")

    st.subheader("Byes y huecos")
    st.caption("`T` titular en la óptima · `B` banca · `bye` · `fuera` (OUT/IR).")
    st.dataframe(out["calendar"], hide_index=True, width="stretch")
    needs = out["needs"]
    if needs.is_empty():
        st.success("Tu roster cubre todos los slots en lo que queda de temporada.")
    else:
        st.warning("**Huecos reales** (slots que tu roster no puede llenar) y el mejor agente libre disponible hoy:")
        st.dataframe(needs, hide_index=True, width="stretch",
                     column_config={"week": "Semana", "esperado": st.column_config.NumberColumn("Esperado", format="%.1f")})

    st.subheader(f"Agentes libres para las próximas {horizon or 4} semanas")
    st.caption("Cuánto suben tus puntos esperados en el horizonte si lo fichas (soltando a quien menos aporte si tu roster "
               "está lleno). `Cubre byes`: semanas en que sería titular mientras uno de tus jugadores descansa.")
    pk = out["pickups"].filter(pl.col("ganancia") > 0).head(12)
    if pk.is_empty():
        st.info("Ningún agente libre mejora tu roster en este horizonte.")
    else:
        st.dataframe(pk.with_columns(pl.col("semanas_titular", "cubre_byes").list.eval(pl.element().cast(pl.Utf8)).list.join(", ")),
                     hide_index=True, width="stretch",
                     column_config={"ganancia": st.column_config.NumberColumn("Ganancia (pts)", format="%+.1f"),
                                    "esperado_horizonte": st.column_config.NumberColumn("Esperado en el horizonte", format="%.1f"),
                                    "semanas_titular": "Semanas titular", "cubre_byes": "Cubre byes"})

# ---------------------------------------------------------------- trades

with tab_trades:
    st.header("Buscador de trades")
    st.caption("Trades 1x1 y 2x1 con los otros 9 equipos en los que **ganan los dos** según el modelo (valor ponderado de la "
               "alineación óptima hasta la semana 17, playoffs ×2, con nivel de reemplazo). Monte Carlo para la incertidumbre y "
               "vista de ESPN para anticipar si el otro manager lo aceptaría.")
    if FINDER_META.exists():
        meta = json.loads(FINDER_META.read_text())
        st.caption(f"Última búsqueda: {meta['when_local']} (semana {meta['week']}).")
    run = st.button("🔍 Buscar trades ahora (~6 min)")
    if run:
        with st.status("Buscando trades…", expanded=True) as status:
            ctx = T.build_trade_context(SEASON, progress=st.write)
            st.write("Evaluando ~30,000 trades posibles…")
            found = T.find_trades(ctx["league_proj"], ctx["fa_proj"], ctx["rules"], ctx["cal"], ctx["cfg"], ctx["me"],
                                  sim=ctx["sim"], verbose=False)
            found["top"].write_parquet(FINDER_PATH)
            FINDER_META.write_text(json.dumps({"when_local": f"{datetime.now(GDL):%d-%m-%Y %H:%M}", "week": ctx["cal"].current_week,
                                               "both_win": found["both"].height}))
            status.update(label="Listo", state="complete")
    if FINDER_PATH.exists():
        top = pl.read_parquet(FINDER_PATH).with_columns(aceptable_segun_espn=pl.col("espn_ellos") >= 0)
        st.dataframe(top.select("equipo", "tipo", "doy", "recibo", "mc_yo", "prob_gano", "mc_ellos", "prob_ganan",
                                "espn_ellos", "aceptable_segun_espn", "suelto_yo", "suelta_ellos"),
                     hide_index=True, width="stretch",
                     column_config={"mc_yo": st.column_config.NumberColumn("Yo (media MC)", format="%+.1f"),
                                    "mc_ellos": st.column_config.NumberColumn("Ellos (media MC)", format="%+.1f"),
                                    "prob_gano": st.column_config.NumberColumn("P(gano)", format="percent"),
                                    "prob_ganan": st.column_config.NumberColumn("P(ganan)", format="percent"),
                                    "espn_ellos": st.column_config.NumberColumn("Ellos según ESPN", format="%+.1f"),
                                    "aceptable_segun_espn": st.column_config.CheckboxColumn("¿Aceptable según ESPN?")})
        st.caption("Con nivel de reemplazo las ganancias suelen ser pequeñas frente al ruido: una P(gano) cercana al 50% es casi una moneda al aire.")
    else:
        st.info("Aún no hay búsquedas guardadas: pulsa el botón.")

# ---------------------------------------------------------------- modelo

with tab_model:
    st.header("¿Qué tan bien va el modelo en 2026?")
    st.caption("Última predicción registrada antes de cada partido frente a los puntos reales (solo jugadores que jugaron), "
               "comparada con la proyección de ESPN sobre los mismos jugadores.")
    ev = load_evaluation()
    if ev.is_empty():
        st.info("Aún no hay semanas registradas con resultado. Aparecerán cuando se jueguen los partidos.")
    else:
        both = ev.filter(pl.col("espn_projection").is_not_null())
        weeks = sorted(both["week"].unique().to_list())
        st.caption(f"Semanas evaluadas: {weeks} · {both.height:,} jugador-semana con proyección de ESPN.")
        mae = (both.group_by("position").agg(pl.len().alias("n"),
                                             (pl.col("pred_model") - pl.col("y")).abs().mean().alias("Modelo"),
                                             (pl.col("espn_projection") - pl.col("y")).abs().mean().alias("ESPN"))
                   .sort("position"))
        total = both.select((pl.col("pred_model") - pl.col("y")).abs().mean(), (pl.col("espn_projection") - pl.col("y")).abs().mean()).row(0)
        m1, m2 = st.columns(2)
        m1.metric("Error medio del modelo (MAE)", f"{total[0]:.2f} pts")
        m2.metric("Error medio de ESPN (MAE)", f"{total[1]:.2f} pts", delta=f"{total[0] - total[1]:+.2f} modelo − ESPN",
                  delta_color="inverse")
        long = mae.unpivot(index=["position", "n"], on=["Modelo", "ESPN"], variable_name="fuente", value_name="mae")
        bars = (alt.Chart(long.to_pandas()).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(x=alt.X("position:N", title=None), xOffset=alt.XOffset("fuente:N", sort=["Modelo", "ESPN"]),
                        y=alt.Y("mae:Q", title="Error medio (puntos)"),
                        color=alt.Color("fuente:N", scale=alt.Scale(domain=["Modelo", "ESPN"], range=[C_MODEL, C_ESPN]),
                                        legend=alt.Legend(title=None, orient="top")),
                        tooltip=[alt.Tooltip("position:N", title="Posición"), alt.Tooltip("fuente:N", title="Fuente"),
                                 alt.Tooltip("mae:Q", title="MAE", format=".2f"), alt.Tooltip("n:Q", title="Jugador-semana")])
                .properties(height=280))
        st.altair_chart(bars, width="stretch", theme="streamlit")
        st.dataframe(mae, hide_index=True, width="stretch",
                     column_config={"Modelo": st.column_config.NumberColumn(format="%.2f"), "ESPN": st.column_config.NumberColumn(format="%.2f")})
        if len(weeks) < 6:
            st.caption("⚠ Con menos de 6 semanas la diferencia no es concluyente.")

        rng = ev.filter(pl.col("pred_q10").is_not_null())
        st.subheader("Cobertura de los rangos P10–P90")
        if rng.is_empty():
            st.info("Aún no hay partidos jugados con rango registrado.")
        else:
            cov = (rng.group_by("position").agg(
                pl.len().alias("n"),
                ((pl.col("y") >= pl.col("pred_q10")) & (pl.col("y") <= pl.col("pred_q90"))).mean().alias("dentro"),
                (pl.col("y") < pl.col("pred_q10")).mean().alias("debajo"),
                (pl.col("y") > pl.col("pred_q90")).mean().alias("encima")).sort("position"))
            st.caption("Objetivo: 80% dentro, 10% debajo y 10% encima.")
            st.dataframe(cov, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="percent") for c in ("dentro", "debajo", "encima")})
