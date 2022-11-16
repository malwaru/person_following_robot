#!/usr/bin/env python3
from geometry_msgs.msg import TransformStamped

import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity


from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

import tf_transformations

import numpy as np


class StaticFramePublisher(Node):
   """
   Broadcast transforms that never change.

   This example publishes transforms from `world` to a static turtle frame.
   The transforms are only published once at startup, and are constant for all
   time.
   """

   def __init__(self,transformation):
      super().__init__('static_tf2_broadcaster')

      self._tf_publisher = StaticTransformBroadcaster(self)
      # Publish static transforms once at startup
      self.make_transforms(transformation)

   def make_transforms(self, transformation):
      static_transformStamped = TransformStamped()
      static_transformStamped.header.stamp = self.get_clock().now().to_msg()
      static_transformStamped.header.frame_id = transformation["parent_frame"]
      static_transformStamped.child_frame_id = transformation["child_frame"]
      static_transformStamped.transform.translation.x = float(transformation["transform"][0])
      static_transformStamped.transform.translation.y = float(transformation["transform"][1])
      static_transformStamped.transform.translation.z = float(transformation["transform"][2])
      quat = tf_transformations.quaternion_from_euler(
            float(transformation["transform"][3]), float(transformation["transform"][4]), float(transformation["transform"][5]))
      static_transformStamped.transform.rotation.x = quat[0]
      static_transformStamped.transform.rotation.y = quat[1]
      static_transformStamped.transform.rotation.z = quat[2]
      static_transformStamped.transform.rotation.w = quat[3]

      self._tf_publisher.sendTransform(static_transformStamped)


def main():
   # logger = rclpy.logging.get_logger('logger')
   rclpy.logging._root_logger.log(
      'Starting static transformer ...',
      LoggingSeverity.INFO
   )
   # obtain parameters from command line arguments
   tansformation = {"parent_frame": "base_link",
                  "child_frame": "camera_depth_optical_frame",
                  "transform": [0, 0, 1, np.deg2rad(-90), 0, 0]}
# pass parameters and initialize node
   rclpy.init()
   node = StaticFramePublisher(tansformation)
   # rclpy.spin(node)
   try:
      rclpy.spin(node)
   except KeyboardInterrupt:
      pass

   rclpy.shutdown()

if __name__ == '__main__':
    main()