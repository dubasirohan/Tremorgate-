#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
source ~/venvs/tremorgate/bin/activate
python -m tremorgate.replay_serial --input "$1" --port /tmp/ttyV0 --rate 100
