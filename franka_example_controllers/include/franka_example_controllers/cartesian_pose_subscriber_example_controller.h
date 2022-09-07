// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <array>
#include <vector>
#include <mutex>
#include <memory>
#include <string>

#include <controller_interface/multi_interface_controller.h>
#include <geometry_msgs/PoseStamped.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>

#include <franka_hw/franka_cartesian_command_interface.h>

namespace franka_example_controllers {

class CartesianPoseSubExampleController
    : public controller_interface::MultiInterfaceController<franka_hw::FrankaPoseCartesianInterface,
                                                            franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  franka_hw::FrankaPoseCartesianInterface* cartesian_pose_interface_;
  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;

  ros::Duration elapsed_time_;
  double filter_params_{0.005};
  std::array<double, 16> initial_pose_{};
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  std::mutex position_and_orientation_d_target_mutex_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;

public: //set to public for testing
  // Basis Function Model
  double nr_dof_{7}; // 3 trans + 4 quaternion
  double nr_basis_fcns_;
  double interval_extension_{0};
  double center_distance_;
  double basis_fcn_width_;
  double nr_time_steps_{1};
  double demo_duration_{30};
  Eigen::VectorXd basis_fcn_centers_;  
  Eigen::VectorXd mean_weights_;
  Eigen::MatrixXd mean_;
  Eigen::MatrixXd basis_fcn_matrix_;
  Eigen::MatrixXd basis_fcn_matrix_dot_;
  Eigen::MatrixXd basis_fcn_matrix_dot_dot_;
  void computeBasisFcns(const ros::Duration& period);
  void computeNextTimeSteps();

    // Goal pose subscriber
  ros::Subscriber sub_goal_pose_;
  void goalPoseCallback(const geometry_msgs::PoseStampedConstPtr& msg);
};

}  // namespace franka_example_controllers
