cmake_minimum_required(VERSION 3.5)
project(person_following_robot)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV 4 REQUIRED)

# Include launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Include launch directory
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Include Cpp "include" directory
include_directories(include)

# Create Cpp executable
add_executable(minimal_cpp_pub src/minimal_cpp_pub.cpp)
ament_target_dependencies(minimal_cpp_pub rclcpp) 
# Install Cpp executables
install(TARGETS
  minimal_cpp_pub
  DESTINATION lib/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/yolo_infer.py
  scripts/person_tracker.py
  scripts/webcam_publisher.py
  scripts/tf_static_transformer.py
  scripts/aruco_detector.py
  scripts/test_script.py
  DESTINATION lib/${PROJECT_NAME}
)

# Instal custom msg and srv

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Object.msg"
  "msg/ObjectList.msg"
  "msg/TrackedObject.msg"
 )

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
