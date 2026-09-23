#!/usr/bin/env bash
# Ejecuta notebooks/04_predicciones.ipynb y hace commit + push del registro de predicciones.
#
# Lo lanza systemd (systemd/fantasy-predictions.timer): miércoles 22:00 y domingo 9:00, hora de Guadalajara.
# Cada ejecución deja una línea en logs/weekly_predictions.log (OK, WARN o ERROR); la salida completa
# de nbconvert y la copia ejecutada del notebook quedan en logs/runs/.
#
# Los domingos (hora de Guadalajara) también calcula la probabilidad de ganar el enfrentamiento de la
# semana (python -m fantasy_ml.matchup --log → data/predictions_log/winprob_<temporada>.csv).
#
# DRY_RUN=1: escribe el registro en una carpeta temporal y prueba el push sin enviar nada.
# FORCE_MATCHUP=1: calcula la probabilidad de ganar aunque no sea domingo (para probar).
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${FANTASY_ML_VENV:-$REPO/../venv}"
LOG_DIR="$REPO/logs"
LOG="$LOG_DIR/weekly_predictions.log"
RUN_DIR="$LOG_DIR/runs"
DRY_RUN="${DRY_RUN:-0}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STEP="inicio"

mkdir -p "$RUN_DIR"
log() { printf '%s [%s] %s\n' "$(date -u +%FT%TZ)" "$1" "$2" | tee -a "$LOG"; }
trap 'rc=$?; log ERROR "falló en el paso \"$STEP\" (código $rc, línea $LINENO): $BASH_COMMAND · detalle: logs/runs/$STAMP.log"; exit $rc' ERR

# Una sola ejecución a la vez
exec 9>"$LOG_DIR/.lock"
if ! flock -n 9; then
    log WARN "otra ejecución sigue en curso; se omite esta"
    exit 0
fi

cd "$REPO"
log START "ejecución $STAMP (dry_run=$DRY_RUN)"

STEP="actualizar repo"
git fetch -q origin
git merge -q --ff-only origin/main

STEP="ejecutar notebook 04"
if [[ "$DRY_RUN" == "1" ]]; then
    export FANTASY_ML_PREDICTIONS_LOG="$(mktemp -d)"
fi
PRED_LOG_DIR="${FANTASY_ML_PREDICTIONS_LOG:-$REPO/data/predictions_log}"
"$VENV/bin/jupyter" nbconvert --to notebook --execute notebooks/04_predicciones.ipynb \
    --ExecutePreprocessor.timeout=1800 --output-dir "$RUN_DIR" --output "04_predicciones_$STAMP" \
    >"$RUN_DIR/$STAMP.log" 2>&1

STEP="resumir"
SUMMARY="$(FANTASY_ML_PREDICTIONS_LOG="$PRED_LOG_DIR" "$VENV/bin/python" - <<'EOF'
import glob, os
import polars as pl
from fantasy_ml import predictions_log as plog
files = sorted(glob.glob(os.path.join(os.environ["FANTASY_ML_PREDICTIONS_LOG"], "[0-9]*.csv")))
if not files:
    print("none")
else:
    season = int(os.path.basename(files[-1]).removesuffix(".csv"))
    log = plog.read(season)
    last = log.filter(pl.col("generated_at_utc") == log["generated_at_utc"].max())
    print(f"{season} {last['week'].max()} {last.height} {last['code_version'][0]}")
EOF
)"
if [[ "$SUMMARY" == "none" ]]; then
    log WARN "no se registró ninguna predicción (¿todos los partidos de la semana ya empezaron?)"
    exit 0
fi
read -r SEASON WEEK ROWS VERSION <<<"$SUMMARY"
[[ "$VERSION" == *-dirty ]] && log WARN "el código tiene cambios sin commit: el registro queda con versión $VERSION"

MATCHUP_MSG=""
if [[ "$(TZ=America/Mexico_City date +%u)" == "7" || "${FORCE_MATCHUP:-0}" == "1" ]]; then
    STEP="probabilidad de ganar"
    FANTASY_ML_PREDICTIONS_LOG="$PRED_LOG_DIR" "$VENV/bin/python" -m fantasy_ml.matchup --log \
        >"$RUN_DIR/$STAMP.matchup.txt" 2>>"$RUN_DIR/$STAMP.log"
    MATCHUP_OUT="$(sed -n 1p "$RUN_DIR/$STAMP.matchup.txt")"
    MATCHUP_MSG="$MATCHUP_OUT"
    log OK "probabilidad de ganar: $MATCHUP_OUT"
fi

STEP="commit y push"
if [[ "$DRY_RUN" == "1" ]]; then
    git push --dry-run -q origin main
    log OK "DRY_RUN: $ROWS predicciones de $SEASON semana $WEEK (versión $VERSION) en $PRED_LOG_DIR; push verificado sin enviar"
    rm -rf "$PRED_LOG_DIR"
    exit 0
fi
if [[ -z "$(git status --porcelain -- data/predictions_log)" ]]; then
    log WARN "el registro no cambió; nada que subir"
    exit 0
fi
LOCAL_TIME="$(TZ=America/Mexico_City date '+%Y-%m-%d %H:%M %Z')"
git add data/predictions_log/*.csv
BODY="$ROWS pre-game predictions generated with code version $VERSION."
[[ -n "$MATCHUP_MSG" ]] && BODY="$BODY
Win probability: $MATCHUP_MSG"
git commit -q -m "Log $SEASON week $WEEK predictions (automatic run, $LOCAL_TIME)" -m "$BODY" -- data/predictions_log/
git push -q origin main

# Las copias ejecutadas del notebook se conservan 60 días
find "$RUN_DIR" -type f -mtime +60 -delete

log OK "$ROWS predicciones de $SEASON semana $WEEK (versión $VERSION); commit $(git rev-parse --short HEAD) subido"
