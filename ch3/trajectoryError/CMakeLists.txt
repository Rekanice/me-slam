cmake_minimum_required( VERSION 2.8 )
project( trajectoryError )

# Sophus
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

#for Pangolin
find_package(Pangolin REQUIRED Eigen REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS} ${Eigen_INCLUDE_DIR})

# Eigen
include_directories("../../libs")

add_executable( trajectoryError ../trajectoryError.cpp )
target_link_libraries(trajectoryError Sophus::Sophus ${Pangolin_LIBRARIES})