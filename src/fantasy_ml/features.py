"""Features sin fuga de información para QB/RB/WR/TE, K y D/ST.

Todas las features de forma reciente usan shift(1) dentro de cada jugador o equipo: la fila de la
semana w solo ve partidos anteriores. Notebook 02 lo verifica con una prueba de fuga.

Las funciones build_* aceptan filas de una semana futura (con estadísticas nulas): sus features se
calculan con los partidos ya jugados y su objetivo `y` queda nulo. Así se predice la semana siguiente.
"""
import polars as pl
from espn_api.football.constant import PRO_TEAM_MAP

from .data import KEYS

POS = ["QB", "RB", "WR", "TE"]
STAT_COLS = ["attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
             "carries", "rushing_yards", "rushing_tds",
             "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
             "target_share", "air_yards_share", "wopr", "fantasy_points_ppr"]
OFF_ROLL = ["fantasy_points_ppr", "xfp", "offense_pct", "attempts", "passing_yards", "passing_tds",
            "carries", "rushing_yards", "targets", "receptions", "receiving_yards", "receiving_air_yards",
            "target_share", "air_yards_share", "wopr"]
COUNT_FEATS = ["games_season", "games_career", "weeks_since_last"]
ROLL_SUFFIXES = ("_l3", "_l5", "_std", "_prev")

OFF_IDS = ["season", "week", "player_id", "player_display_name", "position", "team", "opponent_team", "from_snaps_only"]
K_IDS = ["season", "week", "player_id", "player_display_name", "team", "opponent_team", "espn_id"]
DST_IDS = ["season", "week", "team", "opponent", "espn_id"]

INT_KEYS = [pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32)]
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}


# ---------------------------------------------------------------- IDs

def dst_espn_ids() -> pl.DataFrame:
    """ID de ESPN de cada D/ST (-16000 - proTeamId) con la abreviatura de nflverse."""
    return pl.DataFrame([{"espn_id": -16000 - tid, "team": ESPN_TO_NFLVERSE.get(abbr, abbr)}
                         for tid, abbr in PRO_TEAM_MAP.items() if tid != 0])


def gsis_to_espn(playerids: pl.DataFrame) -> pl.DataFrame:
    return playerids.select(player_id="gsis_id", espn_id="espn_id").drop_nulls().unique("player_id")


# ---------------------------------------------------------------- tablas base

def offense_base(player_stats, snap_counts, opportunity, playerids) -> pl.DataFrame:
    """Una fila por jugador (QB/RB/WR/TE) y partido de temporada regular.

    Incluye los partidos con snaps pero sin estadísticas (0 puntos), que player_stats omite.
    """
    stats_off = (player_stats
        .filter(pl.col("season_type") == "REG", pl.col("position").is_in(POS))
        .select(*INT_KEYS, "player_id", "player_display_name", "position", "team", "opponent_team", *STAT_COLS))

    pfr_to_gsis = playerids.select("pfr_id", "gsis_id").drop_nulls().unique("pfr_id").unique("gsis_id")
    snaps = (snap_counts
        .filter(pl.col("game_type") == "REG", pl.col("position").is_in(POS), pl.col("offense_snaps") > 0)
        .join(pfr_to_gsis, left_on="pfr_player_id", right_on="pfr_id")
        .select(*INT_KEYS, player_id="gsis_id", snap_name="player", snap_position="position",
                snap_team="team", snap_opponent="opponent", offense_pct="offense_pct"))

    xfp = (opportunity
        .filter(pl.col("player_id").is_not_null())
        .select(*INT_KEYS, "player_id", xfp="total_fantasy_points_exp")
        .group_by(*KEYS, "player_id").agg(pl.col("xfp").sum()))

    base = (stats_off
        .join(snaps, on=[*KEYS, "player_id"], how="full", coalesce=True)
        .with_columns(
            from_snaps_only=pl.col("player_display_name").is_null(),
            player_display_name=pl.coalesce("player_display_name", "snap_name"),
            position=pl.coalesce("position", "snap_position"),
            team=pl.coalesce("team", "snap_team"),
            opponent_team=pl.coalesce("opponent_team", "snap_opponent"))
        .drop("snap_name", "snap_position", "snap_team", "snap_opponent")
        .with_columns(pl.col(STAT_COLS).fill_null(0))
        .join(xfp, on=[*KEYS, "player_id"], how="left")
        # sin ninguna oportunidad (0 pases, carries y targets) los puntos esperados son 0
        .with_columns(xfp=pl.when(pl.col("xfp").is_null() & (pl.col("attempts") + pl.col("carries") + pl.col("targets") == 0))
                            .then(0.0).otherwise(pl.col("xfp")))
        .sort("player_id", *KEYS))
    assert base.select("player_id", *KEYS).is_duplicated().sum() == 0, "Filas duplicadas jugador-semana"
    return base


def team_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Una fila por equipo y partido de temporada regular, jugado o no."""
    sched = schedules.filter(pl.col("game_type") == "REG")

    def side(home: bool):
        t, o, sign = ("home", "away", 1) if home else ("away", "home", -1)
        return sched.select(*INT_KEYS, "game_id", "gameday", "gametime",
                            team=pl.col(f"{t}_team"), opponent=pl.col(f"{o}_team"),
                            team_score=pl.col(f"{t}_score"), opp_score=pl.col(f"{o}_score"),
                            is_home=pl.lit(int(home), dtype=pl.Int8), rest_days=pl.col(f"{t}_rest"),
                            spread=sign * pl.col("spread_line"),  # > 0: el equipo es favorito
                            total_line=pl.col("total_line"), roof=pl.col("roof"),
                            temp=pl.col("temp").cast(pl.Float64), wind=pl.col("wind").cast(pl.Float64),
                            div_game=pl.col("div_game").cast(pl.Int8))

    return pl.concat([side(True), side(False)])


def game_context(team_games: pl.DataFrame) -> pl.DataFrame:
    """Líneas de apuestas (de cierre), puntos esperados por equipo, estadio y clima."""
    return (team_games
        .with_columns(implied_team=(pl.col("total_line") + pl.col("spread")) / 2,
                      implied_opp=(pl.col("total_line") - pl.col("spread")) / 2,
                      indoor=pl.col("roof").is_in(["dome", "closed"]).cast(pl.Int8))
        .with_columns(wind=pl.when(pl.col("indoor") == 1).then(0.0).otherwise(pl.col("wind")),
                      temp=pl.when(pl.col("indoor") == 1).then(70.0).otherwise(pl.col("temp")))
        .select(*KEYS, "team", "is_home", "rest_days", "spread", "total_line", "implied_team", "implied_opp",
                "indoor", "temp", "wind", "div_game"))


CONTEXT_FEATS = ["is_home", "rest_days", "spread", "total_line", "implied_team", "implied_opp",
                 "indoor", "temp", "wind", "div_game"]


def team_offense(team_games: pl.DataFrame, team_stats: pl.DataFrame, upcoming: tuple[int, int] | None = None) -> pl.DataFrame:
    """Ofensiva de cada equipo por partido jugado (lo que enfrenta un D/ST).

    Con `upcoming=(season, week)` añade las filas de esa semana sin jugar, para predecirla.
    """
    off = (team_stats.filter(pl.col("season_type") == "REG")
        .select(*INT_KEYS, "team", sacks_allowed="sacks_suffered",
                giveaways=pl.col("passing_interceptions").fill_null(0) + pl.col("sack_fumbles_lost").fill_null(0)
                          + pl.col("rushing_fumbles_lost").fill_null(0) + pl.col("receiving_fumbles_lost").fill_null(0)))
    keep = pl.col("team_score").is_not_null()
    if upcoming:
        keep = keep | ((pl.col("season") == upcoming[0]) & (pl.col("week") == upcoming[1]))
    return (team_games.filter(keep)
        .select(*KEYS, "team", points_scored="team_score")
        .join(off, on=[*KEYS, "team"], how="left"))


# ---------------------------------------------------------------- filas de la semana a predecir

def upcoming_offense(rosters_weekly, team_games, base, season: int, week: int,
                     statuses=("ACT",), require_game: bool = True) -> pl.DataFrame:
    """QB/RB/WR/TE de los rosters de nflverse para esa semana (por defecto: activos y con partido).

    require_game=False incluye a los que no juegan esa semana (bye) con rival nulo: sirve para
    obtener el estado actual de sus features y proyectarlos en semanas posteriores.
    """
    games = team_games.filter(pl.col("season") == season, pl.col("week") == week).select(*KEYS, "team", opponent_team="opponent")
    return (rosters_weekly
        .filter(pl.col("season") == season, pl.col("week") == week, pl.col("status").is_in(list(statuses)),
                pl.col("position").is_in(POS), pl.col("gsis_id").is_not_null())
        .select(*INT_KEYS, player_id="gsis_id", player_display_name="full_name", position="position", team="team")
        .unique(["player_id"])
        .join(games, on=[*KEYS, "team"], how="inner" if require_game else "left")
        .join(base.select("player_id", *KEYS), on=["player_id", *KEYS], how="anti")
        .with_columns(from_snaps_only=pl.lit(False)))


def upcoming_k(rosters_weekly, team_games, points_k, season: int, week: int,
               statuses=("ACT",), require_game: bool = True) -> pl.DataFrame:
    games = team_games.filter(pl.col("season") == season, pl.col("week") == week).select(*KEYS, "team", opponent_team="opponent")
    return (rosters_weekly
        .filter(pl.col("season") == season, pl.col("week") == week, pl.col("status").is_in(list(statuses)),
                pl.col("position") == "K", pl.col("gsis_id").is_not_null())
        .select(*INT_KEYS, player_id="gsis_id", player_display_name="full_name", team="team")
        .unique(["player_id"])
        .join(games, on=[*KEYS, "team"], how="inner" if require_game else "left")
        .join(points_k.select("player_id", *KEYS), on=["player_id", *KEYS], how="anti"))


def upcoming_dst(team_games, points_dst, season: int, week: int, require_game: bool = True) -> pl.DataFrame:
    games = team_games.filter(pl.col("season") == season, pl.col("week") == week).select(*KEYS, "game_id", "team", "opponent")
    if not require_game:  # las 32 defensas; las que descansan quedan con rival nulo
        teams = team_games.filter(pl.col("season") == season).select("team").unique()
        games = teams.with_columns(season=pl.lit(season, pl.Int32), week=pl.lit(week, pl.Int32)).join(
            games, on=[*KEYS, "team"], how="left")
    return games.join(points_dst.select("team", *KEYS), on=["team", *KEYS], how="anti")


# ---------------------------------------------------------------- features

def add_rolling(df, key, cols, windows=(3, 5), prev_season=True) -> pl.DataFrame:
    """Añade medias de partidos ANTERIORES de cada `key` (nunca incluye la fila actual)."""
    key = [key] if isinstance(key, str) else list(key)
    df = df.sort(*key, *KEYS)
    exprs = []
    for c in cols:
        prev = pl.col(c).shift(1)
        exprs += [prev.rolling_mean(w, min_samples=1).over(key).alias(f"{c}_l{w}") for w in windows]
        exprs.append((prev.cum_sum() / prev.cum_count()).over([*key, "season"]).alias(f"{c}_std"))
    df = df.with_columns(exprs)
    if prev_season:
        prev = (df.group_by(*key, "season").agg([pl.col(c).mean().alias(f"{c}_prev") for c in cols])
                  .with_columns(pl.col("season") + 1))
        df = df.join(prev, on=[*key, "season"], how="left")
    return df


def add_counts(df, key) -> pl.DataFrame:
    same_season = pl.col("season") == pl.col("season").shift(1).over(key)
    return df.with_columns(
        games_season=pl.int_range(pl.len()).over([key, "season"]),
        games_career=pl.int_range(pl.len()).over(key),
        weeks_since_last=pl.when(same_season).then(pl.col("week") - pl.col("week").shift(1).over(key)))


def _roll_feats(df) -> list[str]:
    return [c for c in df.columns if c.endswith(ROLL_SUFFIXES)]


def ppr_allowed(base: pl.DataFrame) -> pl.DataFrame:
    """Puntos PPR permitidos por cada defensa a cada posición y partido (nulo si la semana no se ha jugado)."""
    y = pl.col("fantasy_points_ppr")
    return (base.group_by(*KEYS, "opponent_team", "position")
                .agg(opp_ppr_allowed=pl.when(y.is_not_null().any()).then(y.sum()))
                .rename({"opponent_team": "defense"}))


def latest_rolling(df, key, cols, season: int, windows=(5,)) -> pl.DataFrame:
    """Valor que tendrían hoy las features de add_rolling (sin _prev) para el próximo partido de cada `key`.

    Equivale a add_rolling evaluado en una fila nueva al final: media de los últimos w partidos
    (cruzando temporadas) y media de la temporada `season` hasta hoy (nula si aún no jugó en ella).
    """
    key = [key] if isinstance(key, str) else list(key)
    done = df.filter(pl.all_horizontal(pl.col(c).is_not_null() for c in cols)).sort(*key, *KEYS)
    aggs = []
    for c in cols:
        aggs += [pl.col(c).tail(w).mean().alias(f"{c}_l{w}") for w in windows]
        aggs.append(pl.col(c).filter(pl.col("season") == season).mean().alias(f"{c}_std"))
    return done.group_by(key).agg(aggs)


def build_offense(base: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    f = add_counts(add_rolling(base, "player_id", OFF_ROLL), "player_id")

    y = pl.col("fantasy_points_ppr")
    allowed = add_rolling(ppr_allowed(base), ["defense", "position"], ["opp_ppr_allowed"], windows=(5,), prev_season=False)

    f = (f.join(allowed.drop("opp_ppr_allowed"), left_on=[*KEYS, "opponent_team", "position"],
                right_on=[*KEYS, "defense", "position"], how="left")
          .join(context, on=[*KEYS, "team"], how="left")
          .with_columns(y=y))
    return f.select(*OFF_IDS, "y", *_roll_feats(f), *COUNT_FEATS, *CONTEXT_FEATS).sort("player_id", *KEYS)


def build_k(points_k: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    f = add_counts(add_rolling(points_k, "player_id", ["fantasy_points", "fg_att", "fg_made", "pat_att"]), "player_id")
    f = f.join(context, on=[*KEYS, "team"], how="left").with_columns(y=pl.col("fantasy_points"))
    return f.select(*K_IDS, "y", *_roll_feats(f), *COUNT_FEATS, *CONTEXT_FEATS).sort("player_id", *KEYS)


def build_dst(points_dst: pl.DataFrame, team_offense: pl.DataFrame, context: pl.DataFrame) -> pl.DataFrame:
    own = points_dst.with_columns(takeaways=pl.col("def_interceptions") + pl.col("fumble_recovery_opp"))
    f = add_counts(add_rolling(own, "team", ["fantasy_points", "def_sacks", "takeaways", "points_allowed", "yards_allowed"]), "team")
    opp = add_rolling(team_offense, "team", ["points_scored", "sacks_allowed", "giveaways"], windows=(5,), prev_season=False)
    opp_feats = [c for c in opp.columns if c.endswith(("_l5", "_std"))]
    opp = opp.select(*KEYS, pl.col("team").alias("opponent"), *[pl.col(c).alias(f"opp_{c}") for c in opp_feats])
    f = (f.join(opp, on=[*KEYS, "opponent"], how="left")
          .join(context, on=[*KEYS, "team"], how="left")
          .with_columns(y=pl.col("fantasy_points")))
    return f.select(*DST_IDS, "y", *_roll_feats(f), *COUNT_FEATS, *CONTEXT_FEATS).sort("team", *KEYS)


def feature_cols(df: pl.DataFrame) -> list[str]:
    """Columnas de features: todo lo que no es identificador ni objetivo."""
    ids = set(OFF_IDS) | set(K_IDS) | set(DST_IDS) | {"y"}
    return [c for c in df.columns if c not in ids]
