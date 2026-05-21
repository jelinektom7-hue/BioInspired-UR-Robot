# Run Nodes

## Source ROS environment

```bash
. /opt/ros/jazzy/setup.bash && . . ~/ros2_ws/install/setup.bash
```

## Calibrate Camera
Before anything else, you should calibrate your camera. This is done using the "camera-calibration-with-large-chessboards" package.

1. Generate a checkerboard pattern (for example, at https://www.imageonlinetools.com/checkerboard-generator)
2. Take a bunch of images of the calibration checkerboard from various poses and angles.
Please make sure that the ENTIRE photo includes checkerboard pattern, to ensure a robust and accurate calibration.
3. Save the images as PNGs in `~/ros2_ws/src/camera-calibration-with-large-chessboards/input/checkerboard_images`
4. Run the calibration routine using:
```bash
python3 src/camera-calibration-with-large-chessboards/calibration.py -i src/camera-calibration-with-large-chessboards/input/checkerboard_images/ -o src/camera-calibration-with-large-chessboards/output/checkerboard_output/ --debug --kernel_size 25
```

If done correctly, it will output something along the lines of:
```log
Calibration matrix: 
[[1.22109334e+03 0.00000000e+00 6.56137838e+02]
 [0.00000000e+00 1.22328544e+03 4.85644100e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
Distortion parameters (k1, k2, p1, p2, k3):
[[ 1.88988104e-01 -6.15174566e-01 -9.88286870e-05  2.23374393e-04
   5.87519850e-01]]
```

Copy-paste that into the `src/ur5e_whip/image_object_locator/resource/camera_calibration.txt` file, which the image subscriber will use to undistort the incoming images.

## Run Camera Publisher Node

### Option 1: Default settings

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true enable_infra1:=true enable_infra2:=true pointcloud.enable:=true align_depth.enable:=true
```

### Option 2: High Accuracy depth data
```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true enable_infra1:=true enable_infra2:=true pointcloud.enable:=true json_file_path:=src/realsense-ros/realsense2_camera/presets/HighAccuracyPreset.json
```

## Verify camera node working
The camera node should now be running and publishing images to the following topics:

RGB: `/camera/camera/color/image_raw`

Depth: `/camera/camera/depth/image_rect_raw`

Infra1: `/camera/camera/infra1/image_rect_raw`

Infra2: `/camera/camera/infra2/image_rect_raw`

You can verify whether the node is running and publishing images either by checking the publishing frequency of the topics:

```bash
ros2 topic hz <topic>
```

## Run Camera Subscriber Node

```bash
ros2 run image_object_locator image_subscriber
```

This should then subscribe to the appropriate topics, and you should see three OpenCV windows pop up:
1. RGB image feed with ball overlay
2. Segmented image feed
3. Colormapped depth image feed

And then you can test that it is publishing the expected data:
```bash
ros2 topic echo /target_position
```
```bash
ros2 topic echo /target_reference_frame
```

Which should be publishing 3D coordinates, and a flattened 4x4 transformation matrix, respectively.

## Connecting to the UR5E Robot

Install the ur-driver for accessing ros topics on the robot alon with ros controllers:

```bash
sudo apt-get install ros-jazzy-ur ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-jazzy-ros2controlcli
```

Connect your pc and the robot with an ethernet cable.
Turn on the robot's control tablet ad go under settings -> system -> network.
Set the network to have a static address:
- IP Address: 192.168.0.100
- Subnet Mask: 255.255.255.0

On your pc, go to advanced network configuration, edit the Ethernet-network and go to IPV4-Settings.
- Add the IP Address: 192.68.0.77
- Add the netmask: 255.255.255.0

Check that the connection works by pinging the robot:

```bash
ping 192.168.0.100
```

Now run the launch file:
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.0.100
```

And if you want to use mock-up hardware instead run:
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=fake use_mock_hardware:=true
```

Start the UR-cap programme "peterControl" on the polyscope ipad before continuing.

For access to moveit commands run also:
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true
```

For running the server with robot control run: 
```bash
ros2 run cpp_robot_controller controller
```

And to activate the server run the following where the filepath is for your csv file of joint value configurations:

```bash
ros2 service call /whipper cpp_robot_controller/srv/WhipObject "{file_path: <absolute file path>}"
```
An example call could be:

```bash
ros2 service call /whipper cpp_robot_controller/srv/WhipObject "{file_path: /home/peter/ros2_ws/src/ur5e_whip/cpp_robot_controller/trajectory.csv}"
```

# Head back to [README](https://github.com/peterfryd/ur5e_whip/blob/main/README.md#documentation) to finish!
