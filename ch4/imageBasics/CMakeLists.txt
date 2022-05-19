# use c++11 standard
set( CMAKE_CXX_FLAGS "-std=c++11" )
cmake_minimum_required( VERSION 2.8 )
project(imageBasics)

# find OpenCV
find_package( OpenCV REQUIRED )
# include its headers
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( imageBasics imageBasics.cpp )
# link the exe to opencv's libs
target_link_libraries( imageBasics ${OpenCV_LIBS} )

add_executable( undistortImage undistortImage.cpp )
# link the exe to opencv's libs
target_link_libraries( undistortImage ${OpenCV_LIBS} )