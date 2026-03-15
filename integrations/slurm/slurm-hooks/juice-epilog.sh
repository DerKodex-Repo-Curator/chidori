#!/bin/sh
SCRIPT_PATH=$(cd "$(dirname "$0")" && pwd)
. $SCRIPT_PATH/juice-init.sh

exec >> $JUICE_ERROR_LOG 2>&1
chmod 600 $JUICE_ERROR_LOG

if [ -z "$SPANK__SLURM_SPANK_OPTION_juice_juice_pool" ]; then
    exit 0
fi

$JUICE_CMD $JUICE_EXTRA_ARGS --token $JUICE_TOKEN session release --import-session $JUICE_SESSIONS/job-$SLURM_JOBID.cfg
rm -f $JUICE_SESSIONS/job-$SLURM_JOBID.cfg

exit 0