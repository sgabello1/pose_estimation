cmake_minimum_required(VERSION 2.8.3)
project(pose_estimation)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS
  ceres_catkin
  glog_catkin 
  roscpp 
  rospy 
  genmsg 
  cv_bridge
  pcl_conversions
  pcl_ros 
)

find_package(PCL 1.7 REQUIRED)


## Declare a catkin package
catkin_package()

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

add_executable(pose_estimationCeres src/pose_estimationCeres.cpp)
target_link_libraries(pose_estimationCeres ${catkin_LIBRARIES} )


