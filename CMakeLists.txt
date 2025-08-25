cmake_minimum_required(VERSION 3.10.2)
project(road_segmentation_plugin)
set (CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(depthai REQUIRED)
find_package(depthai_ros_driver REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

add_library(
  road_segmentation_plugin SHARED
  src/road_segmentation_plugin.cpp
  src/road_segmentation.cpp
  src/nn_wrapper_road_segmentation.cpp
)

target_include_directories(road_segmentation_plugin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

ament_target_dependencies(road_segmentation_plugin
        depthai
        depthai_ros_driver
        rclcpp
        pluginlib
)

install(TARGETS road_segmentation_plugin
        # EXPORT export_road_segmentation_plugin
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
)

install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

pluginlib_export_plugin_description_file(depthai_ros_driver plugins.xml)

ament_package()
