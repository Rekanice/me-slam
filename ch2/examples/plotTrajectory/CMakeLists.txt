cmake_minimum_required( VERSION 2.8 )
project( plotTrajectory )

#for Pangolin
find_package(Pangolin REQUIRED Eigen REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR})

#for Eigen
#include_directories("../../../libs") #this works too

add_executable(plotTrajectory ../plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})