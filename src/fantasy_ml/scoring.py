"""Puntos de fantasy de K y D/ST con las reglas de config/scoring.yaml.

nflverse no calcula puntos de fantasy para kickers ni defensas. Estas fórmulas se validaron
contra los puntos reales de ESPN en 2026 (notebook 02): K 64/64 partidos, D/ST 63/64.
"""
import polars as pl

INT_KEYS = [pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)]

FG_MADE = {"FG0": ["fg_made_0_19", "fg_made_20_29", "fg_made_30_39"],
           "FG40": ["fg_made_40_49"], "FG50": ["fg_made_50_59"], "FG60": ["fg_made_60_"]}
K_COLS = [c for cols in FG_MADE.values() for c in cols] + ["fg_att", "fg_made", "fg_missed", "fg_blocked",
                                                           "pat_att", "pat_made"]

DST_TS_COLS = ["def_sacks", "def_interceptions", "fumble_recovery_opp", "def_safeties", "def_punt_blocks",
               "def_fg_blocks", "def_pat_blocks", "def_tds", "fumble_recovery_tds", "special_teams_tds",
               "def_2pt_made", "passing_yards", "sack_yards_lost", "rushing_yards"]


def k_points(player_stats: pl.DataFrame, scoring: dict) -> pl.DataFrame:
    """Puntos por partido de kicker (temporada regular).

    ESPN cuenta un FG bloqueado como fallado; en nflverse `fg_missed` no incluye los bloqueados.
    """
    kick = {r["abbr"]: r["points"] for r in scoring["kicking"]}
    return (player_stats
        .filter(pl.col("season_type") == "REG", pl.col("position") == "K")
        .with_columns(pl.col(K_COLS).fill_null(0))
        .with_columns(fantasy_points=(
            sum(pl.sum_horizontal(cols) * kick[abbr] for abbr, cols in FG_MADE.items())
            + (pl.col("fg_missed") + pl.col("fg_blocked")) * kick["FGM"]
            + pl.col("pat_made") * kick["PAT"]))
        .select(*INT_KEYS, "game_id", "player_id", "player_display_name", "team", "opponent_team",
                "fg_att", "fg_made", "fg_missed", "fg_blocked", "pat_att", "pat_made",
                pl.col("fantasy_points").cast(pl.Float64)))


def bucket_points(col: str, rules: list[dict]) -> pl.Expr:
    """Puntos según rangos inclusivos {min, max}; max=None = sin límite superior."""
    expr = pl.lit(None, dtype=pl.Float64)
    for r in reversed(rules):
        in_range = pl.col(col) >= r["min"] if r["max"] is None else pl.col(col).is_between(r["min"], r["max"])
        expr = pl.when(in_range).then(pl.lit(float(r["points"]))).otherwise(expr)
    return expr


def dst_points(team_stats: pl.DataFrame, schedules: pl.DataFrame, scoring: dict) -> pl.DataFrame:
    """Puntos por partido de D/ST (temporada regular, solo partidos jugados).

    Definiciones de ESPN (verificadas en 64/64 partidos de 2026):
    - puntos permitidos = marcador del rival - 6 x (TDs de su defensa + TDs tras recuperar fumble) - 2 x sus safeties
    - yardas permitidas = pase del rival - |yardas perdidas en sacks| + carrera (sack_yards_lost es negativo en nflverse)
    """
    ev = {r["abbr"]: r["points"] for r in scoring["dst"]["events"]}
    ts = (team_stats.filter(pl.col("season_type") == "REG")
          .select("season", "week", "team", *DST_TS_COLS).with_columns(pl.col(DST_TS_COLS).fill_null(0)))

    games = schedules.filter(pl.col("game_type") == "REG", pl.col("home_score").is_not_null())
    sides = pl.concat([
        games.select("season", "week", "game_id", team="home_team", opponent="away_team", opp_score="away_score"),
        games.select("season", "week", "game_id", team="away_team", opponent="home_team", opp_score="home_score"),
    ])
    opp = ts.select("season", "week", opponent="team",
                    opp_pass_yds="passing_yards", opp_sack_yds="sack_yards_lost", opp_rush_yds="rushing_yards",
                    opp_def_tds="def_tds", opp_fr_tds="fumble_recovery_tds", opp_safeties="def_safeties")

    return (sides
        .join(ts.drop("passing_yards", "sack_yards_lost", "rushing_yards"), on=["season", "week", "team"], how="left")
        .join(opp, on=["season", "week", "opponent"], how="left")
        .with_columns(
            points_allowed=pl.col("opp_score") - 6 * (pl.col("opp_def_tds") + pl.col("opp_fr_tds")) - 2 * pl.col("opp_safeties"),
            yards_allowed=pl.col("opp_pass_yds") - pl.col("opp_sack_yds").abs() + pl.col("opp_rush_yds"),
            tds=pl.col("def_tds") + pl.col("fumble_recovery_tds") + pl.col("special_teams_tds"),
            blocks=pl.col("def_punt_blocks") + pl.col("def_fg_blocks") + pl.col("def_pat_blocks"))
        .with_columns(
            pa_points=bucket_points("points_allowed", scoring["dst"]["points_allowed"]),
            ya_points=bucket_points("yards_allowed", scoring["dst"]["yards_allowed"]),
            event_points=(pl.col("def_sacks") * ev["SK"] + pl.col("def_interceptions") * ev["INT"]
                          + pl.col("fumble_recovery_opp") * ev["FR"] + pl.col("def_safeties") * ev["SF"]
                          + pl.col("blocks") * ev["BLKK"] + pl.col("tds") * ev["INTTD"]  # todos los TDs valen 6
                          + pl.col("def_2pt_made") * ev["2PTRET"]))
        .with_columns(fantasy_points=(pl.col("pa_points") + pl.col("ya_points") + pl.col("event_points")).cast(pl.Float64))
        .select(*INT_KEYS, "game_id", "team", "opponent", "points_allowed", "yards_allowed",
                "def_sacks", "def_interceptions", "fumble_recovery_opp", "def_safeties", "blocks", "tds",
                "pa_points", "ya_points", "event_points", "fantasy_points")
        .sort("season", "week", "team"))
