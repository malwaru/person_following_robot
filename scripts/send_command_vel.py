#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from geometry_msgs.msg import Twist

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
            Int32,
            'person_following_robot/recognised_person/data',
            self.tracked_person_callback,
            10)
        self._subscriber_tracked_person
       

        self.publisher_command_vel = self.create_publisher(Twist, 'person_following_robot/command_velocity', 10)

        self._tolerance=[2,10]#Toleance in x and z

        self._vel_msg=Twist()



    def tracked_person_callback(self, msg):
        '''
        Send commands when person is tracked
        '''
        current_position=msg

        linear_diff=self._goal_linear-self._position
        linear_error=np.linalg.norm(linear_diff)  
        angular_error=float(self._goal_angular-self._orientation)
        errooe=math.atan2(self._goal_linear[1],self._goal_linear[0])-self._orientation
        
        if errooe-self._tolerance:
            self.generate_command(msg)
            if abs(angular_error)>self._tolerance_angular:
                self._vel_msg.linear.x=0
                self._vel_msg.angular.z=self._kp_vel_angular*angular_error
                self.send_vel_commands()

            elif abs(angular_error)<self._tolerance_angular and abs(linear_error)>self._tolerance_linear:
                self._vel_msg.linear.x=self._kp_vel_linear*linear_error
                self._vel_msg.angular.z=0
                self.send_vel_commands()

            elif (abs(angular_error)<self._tolerance_angular) and (abs(linear_error)<self._tolerance_linear):
                self._result=self._position
                self._action.set_succeeded(self._result)
                

    def send_vel_commands(self):
        '''
        Generate command
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