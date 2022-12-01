# Person Following Robot 

ROS2 package made to recognize, track and follow a human. 

The person recognition is based on the work done by [1] original repository can be found at https://github.com/meituan/yolov6. In this project we are using a modified version available at https://github.com/malwaru/YOLOv6
The tracking is done by the sort [2] original code available at https://github.com/abewley/sort . In this project we are using a modified version available at https://github.com/malwaru/sort


<img src="https://raw.githubusercontent.com/malwaru/person_following_robot/devel/images/Tracking_pipeline.png" width="400">
## Installation

### Person Following Robot package 

- To install the package specific requirements. Move to the root folder of the package 

```
pip3 install -r requirements.txt
```

The package depends on two other sub packages YoloV6 and Sort both located in the src folder 

For both packages they have to be installed as a python package to do this 
- `cd` into the particular package root folder
- run the command ``` pip install -e . ```
- Then install their particular requirements using `pip3 install -r requirements.txt`


### Note  
Get the YOLOv6 weights from and add to a folder weights 
```
 wget -O yolov6n.pt https://raw.githubusercontent.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n 
 ```

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

- Currently the 



## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status

Currently in active development 

## References

[1] C. Li et al., “YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications.” arXiv, 2022. doi: 10.48550/ARXIV.2209.02976.
[2] A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, “Simple online and realtime tracking,” in 2016 IEEE International Conference on Image Processing (ICIP), 2016, pp. 3464–3468.


