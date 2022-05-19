# use c++11 standard
set( CMAKE_CXX_FLAGS "-std=c++11" )
cmake_minimum_required( VERSION 2.8 )
project(stereo)

# find OpenCV
find_package( OpenCV REQUIRED )
# include its headers
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Pangolin REQUIRED)

add_executable( stereoVision stereoVision.cpp )
# link the exe to opencv's libs
target_link_libraries( stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})