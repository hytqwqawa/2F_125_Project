cmake_minimum_required(VERSION 3.0.2)
project(my_single_vision)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  image_transport
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)