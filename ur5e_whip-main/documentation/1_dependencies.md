# Dependencies

## Dependency 1: Virtual Environment
Before installing python dependencies, setup a virtual environment through VSCode or manually:


### Option #1: VSCode (Easier)

1. Hit `Ctrl + P`, then write `Python: Create Environment`
2. Select `Venv` > `Delete and Recreate` (if it asks) > `Python 3.12.3`
3. Don't install any of the recommended requirements, that tends to break.

### Option #2: Manually (More Control)
```bash
cd ~/ros2_ws && python3 -m venv env
```

To source the virtual environment:
```bash
source ~/ros2_ws/env/bin/activate
```

### Install `requirements.txt`
Then install the python packages in the requirements.txt:
```bash
pip install -r ~/ros2_ws/src/ur5e_whip/requirements.txt
```

To add new files to the requirements.txt, first source your environment and then run the following command:
```bash
pip install <package-name>
```
Then manually add the package to `src/ur5e_whip/requirements.txt`
(Don't use freeze, it adds a lot of random things not needed)

## Dependency 2: Install ROS 2 Jazzy
Install ROS2 Jazzy according to this guide:
[ROS2 Jazzy Installation Guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

## Dependency 3: Install RealSense
### Install Dependencies
```bash
sudo apt update && sudo apt upgrade -y
```
```bash
sudo apt install -y libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev git wget cmake build-essential libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at
```

### Install librealsense2

```bash
git clone https://github.com/IntelRealSense/librealsense.git
```
```bash
cd librealsense && ./scripts/setup_udev_rules.sh
```
```bash
wget https://github.com/IntelRealSense/librealsense/raw/master/scripts/libuvc_installation.sh && chmod +x ./libuvc_installation.sh && ./libuvc_installation.sh
```

Connect a RealSense camera to your machine and run the following to confirm successful installation.

```bash
rs-enumerate-devices -s
```

It should say something like:
```bash
Device Name                   Serial Number       Firmware Version
Intel RealSense D435          815412070596        5.16.0.1
```

### Build librealsense2 SDK

```bash
mkdir build && cd build
```
```bash
cmake ../
```
```bash
sudo make uninstall && make clean && make && sudo make -j$(($(nproc)-1)) install
```

### Install RealSense ROS2 wrapper

Configure your Ubuntu repositories to allow "restricted," "universe," and "multiverse." You can follow the [Ubuntu guide for instructions](https://help.ubuntu.com/community/Repositories/Ubuntu) on doing this. 

```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
```
```bash
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
```
```bash
sudo apt install ros-jazzy-realsense2-*
```
```bash
cd ~/ros2_ws/src/
```
```bash
git clone https://github.com/IntelRealSense/realsense-ros.git -b ros2-master
```

## Dependency 3: Install camera-calibration-with-large-chessboards

Clone the repository in the correct place:

```bash
git clone https://github.com/henrikmidtiby/camera-calibration-with-large-chessboards.git
```

Go to ROS workspace for the next commands

```bash
cd ../
```

## Dependency 4: Initialize ROS Dependency Management

Make sure to update ROS packages:

```bash
sudo apt-get install python3-rosdep -y
```
```bash
sudo rosdep init
```
If you already have a rosdep, that's fine. Continue with the following:
```bash
rosdep update
```
```bash
rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
```

Install additional tools required for building:

```bash
sudo apt install python3-colcon-common-extensions python3-colcon-mixin
```

Add and update the colcon mixin:

```bash
colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml
```
```bash
colcon mixin update default
```

## Dependency 5: Reinforcement Learning Things

We have to install the library ```Gymnasium-Robotics``` from source because of a bug in the pip version.

```
git clone https://github.com/Farama-Foundation/Gymnasium-Robotics.git
cd Gymnasium-Robotics
pip install -e .
```

# Head back to [README](https://github.com/peterfryd/ur5e_whip/blob/main/README.md#step-2-build-the-project) for Step 2!