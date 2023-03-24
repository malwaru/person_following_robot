# Person Following Robot 

ROS2 package made to recognize, track and follow a human. 

The person recognition is based on the work done by [1] original repository can be found at https://github.com/ifzhang/ByteTrack . In this project we are using a modified version available at https://github.com/malwaru/ByteTrack


<img src="https://raw.githubusercontent.com/malwaru/person_following_robot/devel/images/Tracking_pipeline.png" width="400">
## Installation

### Person Following Robot package 

- To install the package specific requirements. Move to the root folder of the package 

```
pip3 install -r requirements.txt
```

The package depends on ByteTrack located in the src folder 

This has to be installed as a python package to do this 
- `cd` into the particular package root folder
- run the command ``` pip install -e . ```
- Then install their particular requirements using `pip3 install -r requirements.txt`

## Quick start

First run the following script to launch the Intel Realsense camera

```
ros2 launch realsense2_camera d400e_rs_launch.py
```

To run the entire pipeline run the following command. 
Before this you will have to activate the virtual envrioment. Which can be done by using the alias 
```
activate_person_following_robot_venv
```
Then to run the launch file
```shell
ros2 launch person_following_robot follow_person.launch.py
```

To record the test data 
- First go to  scripts/person_tracker.py 
- Go to the init function of the trackeByte Node(approximately at line 176)
- In here you have to set the self.record_video=True
- Then in two lines below (approximately line 180) change the file name 
- Now just run the launch file to start camera node then the launch file to start person tracker

If you need to run individual scripts use the following format, <span style="color:red">Note:</span> the namespace argument is important 

`ros2 run person_following_robot <script>  --ros-args -r __ns:=/person_following_robot` 

## To Do

- Load all the topic names from a config file in each script
- Load the frames from handcart description file instead of static transformer
    - Check with romatris_handcart_description packahe to include realsense_description file 
- Create one master package or script to load all packages 



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

[1] Zhang, Yifu / Sun, Peize / Jiang, Yi / Yu, Dongdong / Weng, Fucheng / Yuan, Zehuan / Luo, Ping / Liu, Wenyu / Wang, Xinggang 
ByteTrack: Multi-Object Tracking by Associating Every Detection Box ,2022 Proceedings of the European Conference on Computer Vision (ECCV) 

