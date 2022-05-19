cmake_minimum_required( VERSION 2.8 )
project( useEigen )

# Add header file
include_directories( "../../libs" )  # this works
add_executable( eigenMatrix ../eigenMatrix.cpp )
