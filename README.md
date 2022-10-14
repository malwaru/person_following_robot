# Person Following Robot 

ROS2 package made to recognize, track and follow a human. 

The person recognition is based on the work done by [1] original repository can be found at https://github.com/meituan/yolov6 
The tracking is done by the sort [2] original code available at https://github.com/abewley/sort

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

Get the YOLOv6 weights from and add to a folder weights 
wget -O yolov6n.pt https://raw.githubusercontent.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.

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

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

## References

[1] C. Li et al., “YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications.” arXiv, 2022. doi: 10.48550/ARXIV.2209.02976.
[2] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, “Simple online and realtime tracking,” in 2016 IEEE International Conference on Image Processing (ICIP), 2016, pp. 3464–3468.


