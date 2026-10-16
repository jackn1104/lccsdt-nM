#!/usr/bin/env bash
#SBATCH --job-name=lccsdt
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=460000M
#SBATCH --time=24:00:00
#SBATCH --output=lccsdt-%j.out
#SBATCH --error=lccsdt-%j.err
set -euo pipefail
# Submit from the package root after loading the site's build/runtime modules.
cd "${SLURM_SUBMIT_DIR:?}"
geometry="${1:?Supply an absolute XYZ path}"
basis="${2:?Supply a basis, e.g. aug-cc-pVDZ}"
output="${3:?Supply a NEW absolute output directory}"
exec ./lccsdt run "$geometry" --basis "$basis" --output "$output" --launcher slurm --nodes 2 --ranks-per-node 4 --threads 32 --cpus-per-task 64 --memory-mib 65536 --storage disk --symmetry spin
