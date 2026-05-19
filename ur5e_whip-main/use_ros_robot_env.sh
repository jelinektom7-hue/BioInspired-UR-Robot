#!/bin/bash
deactivate 2>/dev/null || true
export PYTHONNOUSERSITE=1
unset PYTHONPATH
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
echo "Using plain ROS robot env:"
which python3
