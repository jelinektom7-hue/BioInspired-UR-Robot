# Models and stuff


### First training session
The robot had a static initial position, and the target han a static position on the ground. Therefore the model learned to simply lie flat on the ground close to the target, as this often resulted in the tip of the whip being close to the target. The action space was the global angle for every joint

### Second training session:
The initial joint values of the robot were heavlely randomized, and so was the position of the target. The action space was changed to be relative so the robot could change its current angles in the space from -0.1 to 0.1. In this training session the model never got very good at coming close to the target.
We suspect because it was too dificult of a task whith a too big exploration space.

### Third training session
Switched reward function to consider end effector pos, not whip pot. because we want to know if it can learn inverse kinematics at all.  
Did not give good results after 30 min. training.
- tensorboard=reach1

### Fourth training session
Significantly reduced the area the target could be in and switched initual robot joints to be fixed.
The robot sometimes reaches the target and completes a run, otherwise it speed sthrough and tangles in itself.
- tensorboard=reach2

### Fifth training session
Given very slight random variation to initial joint values. No longer stops the environment when the goal is reached. Very promising results, showing that the model should be able to leanr inverse kinematics.
- tensorboard=reach3

### Sixth training session
Switched back to whip. Resets the env when the goal is reached as earlier.
Moved goal further away
- tensorboard=SAC1

### seventh training sesseion
New reward function which now values distance exponetially, having hight velocoity towards the target when close to the target, and rewarding having ever been very close
- tensorboard=SAC2

### 8'th training seesion 
- Reach training
- Model folder = SAC_models_08_reach_04
- tensorboard=SAC3

### 9'th training session 
- Reach training
- Model folder = SAC_models_09_reach_05
- tensorboard=SAC4

### 10'th training session
- Reach training
- tensorboard=SAC5
- Model folder = SAC_models_09_reach_05
- Continue of 9'th


### 11'th
- Whip training
- tensorboard=SAC6
- Model folder = SAC_models_10_whip_02
- Fixed robot and target start positions and orientations