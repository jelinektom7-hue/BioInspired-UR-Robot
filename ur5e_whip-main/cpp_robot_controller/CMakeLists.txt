cmake_minimum_required(VERSION 3.8)
project(cpp_robot_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(moveit_ros_robot_interaction REQUIRED)
find_package(moveit_ros_move_group REQUIRED)
find_package(moveit_core REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rclcpp_action REQUIRED)

# Generate interface
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/WhipObject.srv"
)

# Executable
add_executable(controller src/controller.cpp)
ament_target_dependencies(controller
  rclcpp
  std_msgs
  moveit_ros_planning_interface
  geometry_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  moveit_core
  moveit_ros_move_group
  moveit_ros_robot_interaction
  control_msgs
  rclcpp_action
)

# Install
install(TARGETS
  controller
  DESTINATION lib/${PROJECT_NAME}
)

# Link generated interfaces to the executable
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(controller "${cpp_typesupport_target}")


ament_export_dependencies(rosidl_default_runtime)
ament_package()
