#!/usr/bin/env python3
from geometry_msgs.msg import TransformStamped

import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity


from tf2_ros import TransformBroadcaster

import tf_transformations



class FramePublisher(Node):

    def __init__(self):
        super().__init__('tf2_fake_frame_publisher')

        # Initialize the transform broadcaster
        self.br = TransformBroadcaster(self)
        tf=[1.,1.,0.,0.,0.,0.]

        self.publish_tf(tf)

    def publish_tf(self,tf):
        t = TransformStamped()

        # Read message content and assign it to
        # corresponding tf variables
        # while True:

        rclpy.logging._root_logger.log(
                                    'Starting to publish ...',
                                    LoggingSeverity.INFO
                                )    


        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'

        # Turtle only exists in 2D, thus we get x and y translation
        # coordinates from the message and set the z coordinate to 0
        t.transform.translation.x = tf[0]
        t.transform.translation.y = tf[1]
        t.transform.translation.z = tf[2]

        # For the same reason, turtle can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        q = tf_transformations.quaternion_from_euler(tf[3], tf[4], tf[5])
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the transformation
        self.br.sendTransform(t)


def main():
    rclpy.logging._root_logger.log(
      'Starting fake tf broadcaster ...',
      LoggingSeverity.INFO
   )    
    rclpy.init()
    node = FramePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()