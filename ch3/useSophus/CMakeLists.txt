cmake_minimum_required( VERSION 2.8 )
project( useSophus )

# Sophus
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

# Eigen
include_directories("../../libs")

add_executable( useSophus ../useSophus.cpp )
target_link_libraries(useSophus Sophus::Sophus)