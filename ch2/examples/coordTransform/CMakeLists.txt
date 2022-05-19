cmake_minimum_required( VERSION 2.8 )
project( coordTransform )

# Add header file
include_directories( "../../../libs" )

add_executable( coordinateTransform ../coordinateTransform.cpp )