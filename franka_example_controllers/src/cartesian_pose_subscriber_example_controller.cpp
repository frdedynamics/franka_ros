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
  traj_pub_ = node_handle.advertise<geometry_msgs::PoseStamped>("computed_traj", 1000);
  geometry_msgs::PoseStamped pose_msg;
  traj_pub_.publish(pose_msg);
  ros::spinOnce();

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

  // initialize basis function params
  ROS_INFO_STREAM("NrDOF: " << nr_dof_);
  node_handle.getParam("/promp_params/fixed/nr_basis_fcns", nr_basis_fcns_);
  ROS_INFO_STREAM("NrBasisFcns: " << nr_basis_fcns_);
  node_handle.getParam("/promp_params/learned/mean_demo_duration", demo_duration_);
  ROS_INFO_STREAM("Demo Duration: " << demo_duration_);
  std::vector<double> temp_mean_weights;
  node_handle.getParam("/promp_params/learned/mean_weights", temp_mean_weights);
  mean_weights_ = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(temp_mean_weights.data(), temp_mean_weights.size()); // NB! High chance of error while reading the mean_weights here
  ROS_INFO_STREAM("Mean Weights: " << mean_weights_);
  center_distance_ = 1.0 / (nr_basis_fcns_ - 1 - 2 * interval_extension_); // TODO: some stuff needs to be read from ros param before these lines
  basis_fcn_width_ = 0.5 * pow(center_distance_, 2);
  basis_fcn_centers_.setZero(nr_basis_fcns_);
  for (int i = 0; i < nr_basis_fcns_; i++){
    basis_fcn_centers_(i) = - interval_extension_ * center_distance_ +  i * center_distance_;
  }
  
  return true;
}

void CartesianPoseSubExampleController::starting(const ros::Time& /* time */) {
  // get initial pose 
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_c; // why O_T_EE_d not O_T_EE ?
  ROS_INFO_STREAM("Initial_Pos_y: " << initial_pose_[13]);
  // convert to eigen
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose_.data()));
  // set goal pose to current state
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());
    
/*   Eigen::Matrix4d temp_init_pose;
  temp_init_pose << initial_pose_[0], initial_pose_[1], initial_pose_[2], initial_pose_[3], initial_pose_[4], initial_pose_[5], initial_pose_[6], initial_pose_[7], initial_pose_[8], initial_pose_[9],initial_pose_[10], initial_pose_[11], initial_pose_[12], initial_pose_[13], initial_pose_[14], initial_pose_[15];
  Eigen::Affine3d init_pose(temp_init_pose);
  
  orientation_d_target_ = (Eigen::Quaterniond)init_pose.linear();
  position_d_target_ = init_pose.translation(); */
  //pose should be in eigen vec and quat, not trans mat!

  elapsed_time_ = ros::Duration(0.0);
}

void CartesianPoseSubExampleController::update(const ros::Time& /* time */,
                                            const ros::Duration& period) {
  elapsed_time_ += period;
  CartesianPoseSubExampleController::computeBasisFcns(period);
  CartesianPoseSubExampleController::computeNextTimeSteps();
  position_d_ << mean_.coeff(0, 0), mean_.coeff(0, 1), mean_.coeff(0, 2);
  orientation_d_.coeffs() << mean_.coeff(0, 4), mean_.coeff(0, 5), mean_.coeff(0, 6), mean_.coeff(0, 3);
  
  // Interpolate here within one second from inital_pose to pos_d & ori_d
  double starting_filter_param;
  if ((1.0-elapsed_time_.toSec()) < 0){
    starting_filter_param = 0;
  } else {
    starting_filter_param = 1.0-elapsed_time_.toSec();
  }  
  position_d_ = starting_filter_param * position_d_target_ + (1.0 - starting_filter_param) * position_d_;
  orientation_d_ = orientation_d_.slerp(starting_filter_param, orientation_d_target_);
  
  geometry_msgs::PoseStamped pose_msg;
  pose_msg.header.stamp = ros::Time::now();
  pose_msg.pose.position.x = position_d_.coeff(0);
  pose_msg.pose.position.y = position_d_.coeff(1);
  pose_msg.pose.position.z = position_d_.coeff(2);
    
  pose_msg.pose.orientation.x = orientation_d_.x();
  pose_msg.pose.orientation.y = orientation_d_.y();
  pose_msg.pose.orientation.z = orientation_d_.z();
  pose_msg.pose.orientation.w = orientation_d_.w(); 
  traj_pub_.publish(pose_msg);
    
  Eigen::Affine3d new_transform;
  new_transform.translation() = position_d_;
  new_transform.linear() = orientation_d_.toRotationMatrix();
  Eigen::Matrix4d new_transform_matrix = new_transform.matrix();

  std::array<double, 16> new_pose = initial_pose_;

  //for (size_t i = 0; i < 16; i++) {
    //new_pose[i] = new_transform_matrix(i);
  //}
  new_pose = cartesian_pose_handle_->getRobotState().O_T_EE_c;
  //new_pose[12] = new_transform_matrix(12);
  new_pose[13] = new_transform_matrix(13);
  //new_pose[14] = new_transform_matrix(14);
  //new_pose[13] = new_pose[13] + new_transform_matrix(13)-(-0.216441);
  //new_pose[13] = new_pose[13]+0.00001;
  cartesian_pose_handle_->setCommand(new_pose);
  ROS_INFO_STREAM("Init_Pos_x - Pos_x: " << initial_pose_[12] - new_transform_matrix(12));
  ROS_INFO_STREAM("Init_Pos_y - Pos_y: " << initial_pose_[13] - new_transform_matrix(13));
  ROS_INFO_STREAM("Init_Pos_z - Pos_z: " << initial_pose_[14] - new_transform_matrix(14));

  //ROS_INFO_STREAM("Pos_y: " << new_pose[13]);
  //ROS_INFO_STREAM("Pos_y_d: " << new_transform_matrix(13)-(-0.216441));
  //ROS_INFO_STREAM("elapesd_time: " << elapsed_time_.toSec());
  //ROS_INFO_STREAM("period: " << period.toSec());
  //std::array<double, 7> ddq_d = cartesian_pose_handle_->getRobotState().dq;
  //Eigen::Map<Eigen::VectorXd> ddq_d_vec(&ddq_d[0],7);
  //ROS_INFO_STREAM("ddq_d: " << ddq_d_vec);
 

 //_____________old______________:
  // update parameters changed online through the interactive
  // target by filtering
  //std::lock_guard<std::mutex> position_d_target_mutex_lock(
  //     position_and_orientation_d_target_mutex_);
  // position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  // orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
}


void CartesianPoseSubExampleController::computeBasisFcns(const ros::Duration& period){
  basis_fcn_matrix_.setZero(nr_time_steps_, nr_basis_fcns_);
  basis_fcn_matrix_dot_.setZero(nr_time_steps_, nr_basis_fcns_);
  basis_fcn_matrix_dot_dot_.setZero(nr_time_steps_, nr_basis_fcns_);
  Eigen::VectorXd x_t;
  double elapsed_phase = elapsed_time_.toSec() / demo_duration_;
  double phase_dot = 1 / demo_duration_;
  //ROS_INFO_STREAM("phase_dot: " << phase_dot);
  ROS_INFO_STREAM("elapsed_phase: " << elapsed_phase);
  for (int t = 0; t < nr_time_steps_; t++){
    x_t = elapsed_phase + (t * period.toSec() * phase_dot) - basis_fcn_centers_.array();
    basis_fcn_matrix_.row(t) = Eigen::exp(- x_t.array().square() / (2*basis_fcn_width_));
    //TODO: basis_fcn_matrix_dot_ and basis_fcn_matrix_dot_dot_
  }

  Eigen::MatrixXd basis_fcn_matrix_sum = basis_fcn_matrix_.array().rowwise().sum().inverse();
  basis_fcn_matrix_ = basis_fcn_matrix_.transpose() * basis_fcn_matrix_sum.asDiagonal(); // normalize basis fcns
  Eigen::MatrixXd basis_fcn_matrix_trans = basis_fcn_matrix_.transpose();
  basis_fcn_matrix_ = basis_fcn_matrix_trans;
}

void CartesianPoseSubExampleController::computeNextTimeSteps(){
  mean_.setZero(nr_time_steps_, nr_dof_);
  for (int t = 0; t < nr_time_steps_; t++) {
      Eigen::MatrixXd basis_fcn_matrix_blockdiag_time_sliced = Eigen::MatrixXd::Zero(nr_dof_, basis_fcn_matrix_.cols() * nr_dof_);
      for (int i = 0; i < nr_dof_; i++) {
          basis_fcn_matrix_blockdiag_time_sliced.block(i, i * basis_fcn_matrix_.cols(), basis_fcn_matrix_.row(t).rows(), basis_fcn_matrix_.cols()) = basis_fcn_matrix_.row(t);
      }
      mean_.row(t) = basis_fcn_matrix_blockdiag_time_sliced * mean_weights_;
  }
  //ROS_INFO_STREAM("time steps: " << mean_);
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianPoseSubExampleController,
                       controller_interface::ControllerBase)
