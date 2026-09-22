#!/bin/sh
### One CLI_Train.py run per array element, on an L40S. Submitted by hpc/submit.sh, not bare bsub.
###
###  - Reuses the PLL project's cluster venv (~/PLL_Attempt/.venv): the same pinned packages
###    (torch 2.13.0+cu130, pytorch_optimizer 3.10.1, omegaconf 2.3.1), and home has no room
###    for a second ~3 GB torch install.
###  - gpul40s, not gpuv100: the cu130 wheel has no Volta kernels (PLL hpc/job_sweep_gpu.sh).
###  - The line model is 25k parameters on 5000 windows: 3000 epochs is minutes, not hours.
###    -W 1:00 keeps a wide margin and lets LSF backfill the jobs between long ones.
###  - A GPU run is a fresh draw, not a replica of the laptop (MPS) seed with the same number:
###    compare GPU runs with GPU runs only. That is why each exp file names its own results_dir.
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 1:00
#BSUB -o logs/%J_%I.out
#BSUB -e logs/%J_%I.err

cd "$LS_SUBCWD" || exit 1
. "$HOME/PLL_Attempt/.venv/bin/activate"

export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

: "${CONFIGS:?CONFIGS not set -- submit via hpc/submit.sh, not bare bsub}"

LINE=$(sed -n "${LSB_JOBINDEX}p" "$CONFIGS")
[ -n "$LINE" ] || { echo "no config on line $LSB_JOBINDEX of $CONFIGS"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "cmd  python src/CLI_Train.py $LINE"
# shellcheck disable=SC2086
exec python src/CLI_Train.py $LINE
