#!/bin/bash
deactivate 2>/dev/null || true
source /home/dragos/ros2_ws/src/BioInspired-UR-Robot/ur5e_whip-main/mujoco_env/bin/activate
export PYTHONNOUSERSITE=1
unset PYTHONPATH
echo "Using RL/MuJoCo/SAC env:"
which python
python - <<'PY'
import numpy as np
print("NumPy:", np.__version__)
PY
