#!/usr/bin/env python3

from time import sleep
import rclpy
from rclpy.node import Node
from person_following_robot.msg import TrackedObject
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import Twist,PoseStamped,Pose
import numpy as np

###########################################################################
###########################################################################
###########################################################################
## TO DO:
## 1. Convert to PID commands
## 2. Get depth data 
## 3. 
#############################################################################
#############################################################################
#############################################################################

class SendCommandVel(Node):

    def __init__(self):
        super().__init__('person_tracker')
        #Subscribe to the bounding box values from the person 
        self._subscriber_tracked_person = self.create_subscription(
            TrackedObject,
            'person_following_robot/recognised_person/data',
            self.tracked_person_callback,
            10)
        self._subscriber_tracked_person

        self.publisher_command_vel = self.create_publisher(Twist, 'person_following_robot/command_velocity', 10)

        self._tolerance=[0.2,10]#Toleance in theta and z

        self._vel_msg=Twist()

        ## PID constants 
        self.angular_PID=[0.1,0.1,0.1]
        self.linear_PID =[0.1,0.1,0.1]

        ## TF2 listerners for frame transform
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)



    def tracked_person_callback(self, msg):
        '''
        Send commands when person is tracked
        '''
        if msg.success:
            try:
                input_pose = Pose()
                pose_stamped=PoseStamped()
                input_pose.position.x = msg.position[0]
                input_pose.position.y = msg.position[1]
                input_pose.position.z = msg.position[2]
                input_pose.orientation.x = 0.
                input_pose.orientation.y = 0.
                input_pose.orientation.z = 0.
                input_pose.orientation.w = 0.
                pose_stamped.pose = input_pose
                pose_stamped.header.frame_id = 'base_link'
                pose_stamped.header.stamp = rclpy.time.Time()            
                trans_pose = self.tf_buffer.transform(pose_stamped,'base_link')
                delta_x=trans_pose.pose.position.x
                delta_y=trans_pose.pose.position.z

            except:
                self.get_logger().info(f"Cannot get TF transform")



            delta_x=0
            delta_z=0

            # linear_diff=self._goal_linear-self._position
            linear_error=delta_x
            angular_error=np.arctan(delta_x/delta_z)

            ## If linear error or angular error is higer than tolerance 
            if (linear_error>self._tolerance[1]) or (angular_error>self._tolerance[0]):
                self.generate_command(angular_error,linear_error)
                self.send_vel_commands()
            


    def generate_command(self,error_a,error_l):
        '''
        Generate PID commands

        Parameters
        -----------
        error_a : angular error in radians
        error_l : linear error in meters
        '''
        angular_command=np.linalg.norm(error_a*self.angular_PID)
        linear_command = np.linalg.norm(error_l*self.linear_PID)
        self._vel_msg.linear.x=linear_command
        self._vel_msg.angular.z=angular_command

                    

    def send_vel_commands(self):
        '''
        Send commands
        '''
        self.publisher_command_vel.publish(self._vel_msg)
        


 

    

def main(args=None):
    rclpy.init(args=args)
    person_tracker = SendCommandVel()
    rclpy.spin(person_tracker)
    # Destroy the node explicitly  
    person_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()