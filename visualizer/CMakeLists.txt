cmake_minimum_required(VERSION 3.15)
project(book_pipeline)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_BUILD_TYPE Release)

find_package(Ceres REQUIRED)

#if(NOT Ceres_FOUND)  # Looks like Ceres find glog on the own, so separate search isn't necessary
#    message("CERES NOT FOUND ------------------------------------------------------------")
#else()
#    message("CERES FOUND ------------------------------------------------------------")
#endif()

#set(OpenCV_DIR "" CACHE PATH "~/install/opencv-4.2.0/build_4")
find_package(OpenCV 4 REQUIRED core calib3d features2d sfm viz)
find_package(OpenMVS REQUIRED)
find_package(Boost REQUIRED filesystem graph)

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenMVS_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})
message(STATUS ${OpenCV_LIBRARIES} ${OpenMVS_LIBRARIES})


#add_executable(book_pipeline ../extra/book_pipeline.cpp)
#target_link_libraries(book_pipeline
#        ${OpenCV_LIBRARIES}
#        ${Boost_LIBRARIES}
#        #        ${OpenMVS_LIBRARIES}
#        )

add_executable(visualizer visualizer.cpp)
target_link_libraries(visualizer
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
        #        ${OpenMVS_LIBRARIES}
        )