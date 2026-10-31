#!/bin/bash
# Submit a WorldScore benchmark run as a Slurm job on NCSA Delta.
#
# Usage, from the repo root on Delta, after setup_delta.sh:
#   bash tools/eval/worldscore/submit_delta.sh <run name> [run_small.sh|run.sh] [config] [overrides...]
#
#   bash tools/eval/worldscore/submit_delta.sh 2026-11-01-ws-minwm-s1
#       the small benchmark (16 instances), original minWM Stage 1
#   NGPU=4 TIME=12:00:00 bash tools/eval/worldscore/submit_delta.sh 2026-11-01-ws-minwm-s1 run.sh \
#       configs/wan21/action2v/infer/stage1_ar_tf.py
#       grow the same run to the full subset (121 instances)
#
# Everything lands in $WORK/outputs/WaveFront/<run name>/: summary.md, the
# instance dirs with videos, slurm-<job id>.out, and job-<job id>.sbatch (the
# script submitted). Resubmitting the same run name resumes where it stopped;
# with the default preemptible partition the job is also requeued on preemption.
#
# Environment (defaults in brackets): WORK (required), ACCOUNT [bjki-delta-gpu],
# PARTITION [gpuA40x4-preempt], NGPU [1], TIME [04:00:00].
set -eo pipefail

usage() { sed -n '2,20p' "$0"; exit 2; }
[ $# -ge 1 ] || usage
run=$1
shift
script=run_small.sh
if [ $# -ge 1 ] && [[ $1 == run_small.sh || $1 == run.sh ]]; then
  script=$1
  shift
fi
[[ $run =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run name: letters, digits and . _ - only" >&2; exit 2; }
: "${WORK:?set WORK, e.g. export WORK=/work/hdd/<project>/\$USER}"
account=${ACCOUNT:-bjki-delta-gpu}
partition=${PARTITION:-gpuA40x4-preempt}
ngpu=${NGPU:-1}
walltime=${TIME:-04:00:00}

repo=$(cd "$(dirname "$0")/../../.." && pwd)
venv=$WORK/venvs/WaveFront
out=$WORK/outputs/WaveFront/$run
[ -x "$venv/bin/python" ] || { echo "no $venv; run tools/eval/worldscore/setup_delta.sh first" >&2; exit 1; }
if [ -n "$(git -C "$repo" status --porcelain --untracked-files=no)" ]; then
  echo "warning: uncommitted changes in $repo; the job runs them as they are" >&2
fi
mkdir -p "$out"
args=
[ $# -eq 0 ] || args=$(printf '%q ' "$@")

job=$(mktemp)
trap 'rm -f "$job"' EXIT
{
  echo '#!/bin/bash'
  echo "#SBATCH --job-name=$run"
  echo "#SBATCH --account=$account"
  echo "#SBATCH --partition=$partition"
  echo "#SBATCH --nodes=1"
  echo "#SBATCH --ntasks-per-node=1"
  echo "#SBATCH --gpus-per-node=$ngpu"
  echo "#SBATCH --cpus-per-task=$((16 * ngpu))"
  echo "#SBATCH --mem=$((60 * ngpu))g"
  echo "#SBATCH --time=$walltime"
  echo "#SBATCH --output=$out/slurm-%j.out"
  if [[ $partition == *preempt* ]]; then echo "#SBATCH --requeue"; echo "#SBATCH --open-mode=append"; fi
  cat <<EOF
set -eo pipefail
cd $(printf %q "$repo")
echo "job \$SLURM_JOB_ID on \$(hostname), commit \$(git rev-parse --short HEAD), \$(date)"
source $(printf %q "$venv/bin/activate")
export WORK=$(printf %q "$WORK") OUT_DIR=$(printf %q "$out") TOKENIZERS_PARALLELISM=false
bash tools/eval/worldscore/$script $args
EOF
} >"$job"

jobid=$(sbatch --parsable "$job")
jobid=${jobid%%;*}
cp "$job" "$out/job-$jobid.sbatch"
echo "submitted $run as job $jobid ($script, $ngpu GPU, $partition, $walltime)"
echo "  log:     $out/slurm-$jobid.out"
echo "  results: $out/summary.md"
