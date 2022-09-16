# # If needed to load from yaml file 
# import os
# from ament_index_python.packages import get_package_share_directory


###########################################################################
###########################################################################
###########################################################################
## TO DO:
## 1. Load data from yaml files
## 2. Start the tracker when Yolo data is available not with a timer

#############################################################################
#############################################################################
#############################################################################


from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():

    ld =LaunchDescription()

    realsense_camera=Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        parameters=[{'depth_align' :True}],
        namespace="camera"
        

    )
    

    tf_broadcaster=Node(
        package="person_following_robot",
        executable="tf_static_transformer.py",
        namespace="person_following_robot"
        
    )

    yolo_infer=Node(
        package="person_following_robot",
        executable="yolo_infer.py",
        namespace="person_following_robot"

    )

    aruco_detector=Node(
        package="person_following_robot",
        executable="aruco_detector.py",
        namespace="person_following_robot"

    )

    person_tracker=TimerAction(period=10.0,actions=[Node(
        package="person_following_robot",
        executable="person_tracker.py",
        namespace="person_following_robot"
    )])

    # ld.add_action(realsense_camera)
    ld.add_action(tf_broadcaster)
    ld.add_action(aruco_detector)
    ld.add_action(yolo_infer)
    # ld.add_action(person_tracker)



#     # Example of loading from yaml file 
#     config = os.path.join(
#     get_package_share_directory('ros2_tutorials'),
#     'config',
#     'params.yaml'
#     )

#     node=Node(
#     package = 'ros2_tutorials',
#     name = 'your_amazing_node',
#     executable = 'test_yaml_params',
#     parameters = [config]
# )
        

    return ld