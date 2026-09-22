#!/bin/sh
# Submit one LSF array job, one element per config line (run from the project root):
#
#     sh hpc/submit.sh hpc/exp01_comb.txt comb
#
# Ported from PLL_Attempt/hpc/submit.sh. Comments and blank lines are stripped into
# hpc/.expanded/ and the [1-N] range is taken from THAT file, so the range and the line
# numbers the job reads can never disagree. The config path is BAKED INTO a generated copy
# of the jobscript: DTU's LSF does not forward the submitting shell's environment.
#
# Before anything is queued, every line is checked twice on the login node:
#   1. CLI_Train.py <line> list=true  -- a typo'd key fails struct mode here, not N times in the queue
#   2. every dataset the lines name (or the config default) must exist AND open
set -eu

SRC=${1:?usage: sh hpc/submit.sh <configs.txt> [jobname] [jobscript]}
NAME=${2:-line}
SCRIPT=${3:-hpc/job_train_gpu.sh}
PY=${PY:-$HOME/PLL_Attempt/.venv/bin/python}

[ -f "$SRC" ]    || { echo "no such config file: $SRC"; exit 1; }
[ -f "$SCRIPT" ] || { echo "no such jobscript: $SCRIPT"; exit 1; }
[ -x "$PY" ]     || { echo "no python at $PY -- set PY=... or build the PLL venv first"; exit 1; }

mkdir -p logs hpc/.expanded
EXP="hpc/.expanded/$(basename "$SRC")"
sed -e 's/[[:space:]]*#.*$//' -e '/^[[:space:]]*$/d' "$SRC" > "$EXP"
N=$(wc -l < "$EXP" | tr -d ' ')
[ "$N" -gt 0 ] || { echo "no configs in $SRC"; exit 1; }

# 1. every line parses against config/Line_Training.yml (no torch import, fast)
while IFS= read -r LINE; do
    # shellcheck disable=SC2086
    "$PY" src/CLI_Train.py $LINE list=true > /dev/null \
        || { echo "refusing to submit: bad config line: $LINE"; exit 1; }
done < "$EXP"

# 2. every dataset exists and opens (a file killed mid-save stats fine and fails on load)
"$PY" - "$EXP" <<'PYEOF'
import sys, zipfile
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
base = OmegaConf.load("config/Line_Training.yml")
bad = set()
for line in open(sys.argv[1]):
    cfg = OmegaConf.merge(base, OmegaConf.from_dotlist(line.split()))
    for key in ("dataset", "eval_dataset"):
        p = Path("data") / cfg[key]
        try:
            np.load(p)
        except FileNotFoundError:
            bad.add(f"{p}: missing")
        except zipfile.BadZipFile:
            bad.add(f"{p}: truncated or still being written")
if bad:
    print("refusing to submit:\n    " + "\n    ".join(sorted(bad)))
    sys.exit(1)
PYEOF

# generated jobscript with the config path written in
JOB="hpc/.expanded/$(basename "$SRC" .txt)__$(basename "$SCRIPT")"
sed "s|^: \"\\\${CONFIGS:?.*|CONFIGS=\"$EXP\"|" "$SCRIPT" > "$JOB"
grep -q "^CONFIGS=" "$JOB" || { echo "could not inject CONFIGS into $SCRIPT"; exit 1; }

echo "$SRC -> $N jobs via $SCRIPT   (config path baked into $JOB)"
nl -ba "$EXP"
echo

bsub -J "${NAME}[1-${N}]" < "$JOB"

echo
echo "watch with:  bjobs -w          (bjobs -A drops finished elements from its summary)"
echo "kill  with:  bkill -J ${NAME}"
echo "logs:        logs/<jobid>_<index>.out   (.err for errors)"
