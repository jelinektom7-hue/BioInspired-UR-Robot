#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include "rclcpp/rclcpp.hpp"
#include "cpp_robot_controller/srv/whip_object.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2/LinearMath/Quaternion.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include <fstream>
#include <sstream>
#include <memory>
#include <vector>
#include <string>

using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
using GoalHandleFollowJointTrajectory = rclcpp_action::ClientGoalHandle<FollowJointTrajectory>;

class Whipper : public rclcpp::Node
{
public:
  Whipper() : Node("whipper")
  {
    action_client_ = rclcpp_action::create_client<FollowJointTrajectory>(
      this, "/scaled_joint_trajectory_controller/follow_joint_trajectory");

    service_ = this->create_service<cpp_robot_controller::srv::WhipObject>(
      "whipper", std::bind(&Whipper::move, this, std::placeholders::_1, std::placeholders::_2));
  }

  void move(const std::shared_ptr<cpp_robot_controller::srv::WhipObject::Request> request,
            std::shared_ptr<cpp_robot_controller::srv::WhipObject::Response> response)
  {
    if (!action_client_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available after waiting");
      response->success = false;
      return;
    }

    auto trajectory_data = loadCSV(request->file_path);
    if (trajectory_data.empty() || trajectory_data[0].size() != 7) {
      RCLCPP_ERROR(this->get_logger(), "CSV must contain 6 joint values + 1 time per line.");
      response->success = false;
      return;
    }

    auto goal_msg = FollowJointTrajectory::Goal();

    goal_msg.trajectory.joint_names = {
      "shoulder_pan_joint",
      "shoulder_lift_joint",
      "elbow_joint",
      "wrist_1_joint",
      "wrist_2_joint",
      "wrist_3_joint"
    };

    for (const auto &row : trajectory_data) {
      trajectory_msgs::msg::JointTrajectoryPoint point;
      point.positions = std::vector<double>(row.begin(), row.begin() + 6);
      point.time_from_start = rclcpp::Duration::from_seconds(row[6]);
      goal_msg.trajectory.points.push_back(point);
    }

    auto send_goal_options = rclcpp_action::Client<FollowJointTrajectory>::SendGoalOptions();

    // **Corrected Goal Response Callback signature**
    send_goal_options.goal_response_callback =
      [this](std::shared_ptr<GoalHandleFollowJointTrajectory> goal_handle) {
        if (!goal_handle) {
          RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
        } else {
          RCLCPP_INFO(this->get_logger(), "Goal accepted by server");
        }
      };

    send_goal_options.result_callback =
      [this, response](const GoalHandleFollowJointTrajectory::WrappedResult &result) {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
          RCLCPP_INFO(this->get_logger(), "Trajectory execution succeeded");
          response->success = true;
        } else {
          RCLCPP_ERROR(this->get_logger(), "Trajectory execution failed with code %d", static_cast<int>(result.code));
          response->success = false;
        }
      };

    action_client_->async_send_goal(goal_msg, send_goal_options);
  }

private:
  rclcpp_action::Client<FollowJointTrajectory>::SharedPtr action_client_;
  rclcpp::Service<cpp_robot_controller::srv::WhipObject>::SharedPtr service_;

  std::vector<std::vector<double>> loadCSV(const std::string &file_path)
  {
    std::ifstream file(file_path);
    std::vector<std::vector<double>> data;

    if (!file.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", file_path.c_str());
      return data;
    }

    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string value;
      std::vector<double> row;

      while (std::getline(ss, value, ',')) {
        try {
          row.push_back(std::stod(value));
        } catch (...) {
          RCLCPP_WARN(this->get_logger(), "Invalid value in CSV.");
        }
      }
      if (row.size() == 7) // 6 joints + 1 time
        data.push_back(row);
    }
    return data;
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Whipper>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
