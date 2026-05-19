#!/bin/bash
deactivate 2>/dev/null || true
source /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/ros_camera_env/bin/activate
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
echo "Using ROS camera/cv_bridge env:"
which python
python - <<'PY'
import numpy as np
print("NumPy:", np.__version__)
PY
