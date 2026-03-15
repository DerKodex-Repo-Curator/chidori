#!/bin/sh
SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)
. $SCRIPT_PATH/juice-init.sh

exec >> $JUICE_ERROR_LOG 2>&1
chmod 600 $JUICE_ERROR_LOG

if [ -z "$SPANK__SLURM_SPANK_OPTION_juice_juice_pool" ]; then
    exit 0
fi

NUM_GPUS=""
if [ ! -z "$SPANK__SLURM_SPANK_OPTION_juice_juice_num_gpus" ]; then
    NUM_GPUS="--num-gpus $SPANK__SLURM_SPANK_OPTION_juice_juice_num_gpus"
fi

VRAM=""
if [ ! -z "$SPANK__SLURM_SPANK_OPTION_juice_juice_vram" ]; then
    VRAM="--vram $SPANK__SLURM_SPANK_OPTION_juice_juice_vram"
fi

mkdir -p $JUICE_SESSIONS
$JUICE_CMD $JUICE_EXTRA_ARGS --token $JUICE_TOKEN session request --export-session $JUICE_SESSIONS/job-$SLURM_JOBID.cfg --pool-ids $SPANK__SLURM_SPANK_OPTION_juice_juice_pool $NUM_GPUS $VRAM
RESULT=$?
exit $RESULT
