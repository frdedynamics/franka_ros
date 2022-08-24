// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_pose_subscriber_example_controller.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <numeric>

#include <controller_interface/controller_base.h>
#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

namespace franka_example_controllers {

bool CartesianPoseSubExampleController::init(hardware_interface::RobotHW* robot_hardware,
                                          ros::NodeHandle& node_handle) {
  sub_goal_pose_ = node_handle.subscribe(
      "goal_pose", 20, &CartesianPoseSubExampleController::goalPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  cartesian_pose_interface_ = robot_hardware->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianPoseSubExampleController: Could not get Cartesian Pose "
        "interface from hardware");
    return false;
  }

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("CartesianPoseSubExampleController: Could not get parameter arm_id");
    return false;
  }

  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface_->getHandle(arm_id + "_robot"));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseSubExampleController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

  auto state_interface = robot_hardware->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR("CartesianPoseSubExampleController: Could not get state interface from hardware");
    return false;
  }

  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseSubExampleController: Exception getting state handle: " << e.what());
    return false;
  }

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  return true;
}

void CartesianPoseSubExampleController::starting(const ros::Time& /* time */) {
  // get initial pose 
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE; // why O_T_EE_d not O_T_EE ?
  // convert to eigen
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose_.data()));
  // set goal pose to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());
  // initialize basis function params
  center_distance_ = 1.0 / (nr_basis_fcns_ - 1 - 2 * interval_extension_);
  basis_fcn_width_ = 0.5 * pow(center_distance_, 2);
  std::vector<int> temp_range(nr_basis_fcns);
  basis_fcn_centers_ = - interval_extension_ * center_distance_ + std::iota(std::begin(temp_range), std::end(temp_range), 0) * center_distance_;
  elapsed_time_ = ros::Duration(0.0);
}

void CartesianPoseSubExampleController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {
  elapsed_time_ += period;

  Eigen::Affine3d new_transform;
  new_transform.translation() = position_d_;
  new_transform.linear() = orientation_d_.toRotationMatrix();
  Eigen::Matrix4d new_transform_matrix = new_transform.matrix();

  std::array<double, 16> new_pose = initial_pose_;

  for (size_t i = 0; i < 16; i++) {
    new_pose[i] = new_transform_matrix(i);
  }
  cartesian_pose_handle_->setCommand(new_pose);

  // update parameters changed online through the interactive
  // target by filtering
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}

void CartesianPoseSubExampleController::goalPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      msg->pose.orientation.z, msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianPoseSubExampleController,
                       controller_interface::ControllerBase)
