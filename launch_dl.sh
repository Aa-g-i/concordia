#!/bin/bash

# Simple Launcher for deploying a DeliberateLab export sequentially onto Concordia models.

export PYTHONPATH=$PWD
echo "> Booting Concordia Orchestrator Pipeline..."

if [ "$#" -lt 1 ]; then
    echo "Usage: ./launch_dl.sh <path_to_zip> [cohort_id] [comma_separated_stages]"
    echo ""
    echo "Examples:"
    echo "  Run an entire experiment chronologically: ./launch_dl.sh /path/to/dl_data.zip"
    echo "  Run a specific cohort: ./launch_dl.sh /path/to/dl_data.zip mixk3s38-ey2c1o"
    echo "  Select specific stages: ./launch_dl.sh /path/to/data.zip mixk3s38-ey2c1o discussion-round-1,discussion-round-3"
    exit 1
fi

ZIP_PATH=$1
COHORT_ARG=""
STAGES_ARG=""

if [ "$#" -ge 2 ]; then
    COHORT_ARG="--cohort $2"
fi

if [ "$#" -ge 3 ]; then
    STAGES_ARG="--stages $3"
fi

python concordia/deliberate_integration/run_dl_experiment.py --zip "$ZIP_PATH" $COHORT_ARG $STAGES_ARG
