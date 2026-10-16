#!/usr/bin/env bash
#SBATCH --job-name=lccsdt-restart
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=460000M
#SBATCH --time=24:00:00
#SBATCH --output=lccsdt-restart-%j.out
#SBATCH --error=lccsdt-restart-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
run="${1:?Supply the complete saved run directory}"
# Match this allocation to the original run.json; no settings are overridden.
exec ./lccsdt restart "$run" --triples-only
