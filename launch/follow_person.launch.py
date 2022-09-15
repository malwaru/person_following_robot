# # If needed to load from yaml file 
# import os
# from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    '''
    To do:

    Run the tracker node when yolo data is available instead of a timer
    
    
    '''
    ld =LaunchDescription()
    # add arrgument "namespace" and "name" if neccessary
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


    ld.add_action(tf_broadcaster)
    ld.add_action(yolo_infer)
    ld.add_action(person_tracker)
    ld.add_action(aruco_detector)



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