# Person Following Robot 

ROS2 package made to recognize, track and follow a human. 

The person recognition is based on the work done by [1] original repository can be found at https://github.com/meituan/yolov6 
This package is intended for a robot to recognize, track and follow a human. 

Get tge YOLOv6 
wget -O publisher_member_function.cpp https://raw.githubusercontent.com/ros2/examples/foxy/rclcpp/topics/minimal_publisher/member_function.cpp
https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.pt
## Quick start

First run the following script to launch the Intel Realsense camera

```
ros2 run realsense2_camera realsense2_camera_node --ros-args -p align_depth:=true -p  spatial_filter.enable:=true -p temporal_filter.enable:=true
```

To run the entire pipeline run the following command

```shell
ros2 launch person_following_robot follow_person
```

If you need to run individual scripts use the following format, <span style="color:red">Note:</span> the namespace argument is important 

`ros2 run person_following_robot <script>  --ros-args -r __ns:=/person_following_robot` 

## References

[1] C. Li et al., “YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications.” arXiv, 2022. doi: 10.48550/ARXIV.2209.02976.