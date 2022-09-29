#!/usr/bin/env python3
import imp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import cv2
from rclpy.logging import LoggingSeverity
import pyrealsense2 as rs
import numpy as np

###########################################################################
# TO DO:
# 1. Create a new msg type to send tracking success
# 2. Use a gaussian filter to cut out the anamolies and get average depth
# 3. If QR code is not available find the largest box
#############################################################################


class CamStream(Node):

    def __init__(self, device=0):
        super().__init__('person_tracker')
        # self.publisher_webstream = self.create_publisher(Image,
        #                                                 '/aligned_depth_to_color/image_raw',
        #                                                 10)

        self._subscriber_camera_depth_aligned = self.create_subscription(
                                            Image,
                                            '/aligned_depth_to_color/image_raw',
                                            self.camera_depth_aligned_callback,
                                            10)

        self._subscriber_camera_raw = self.create_subscription(
                                        Image,
                                        '/color/image_raw',
                                        self.camera_raw_callback,
                                        10)
        # Other variables
        self._cvbridge=CvBridge()

    # def camera_depth_aligned_callback(self,msg):
    #     '''
    #     '''
    #     depth_stream=self._cvbridge.imgmsg_to_cv2(msg)
    #     x=300.
    #     y=400.
    #     px=np.array([300.0,400.0])
    #     depth=30.
    #     world_coordinate=rs.rs2_deproject_pixel_to_point(px,depth)
    #     self.get_logger().info(f"Received depth {world_coordinate}")   

    #     cv2.imshow("Depth",depth_stream)
    #     cv2.waitKey(1)

    def camera_depth_aligned_callback(self,msg):
        '''
        '''
        depth_stream=self._cvbridge.imgmsg_to_cv2(msg)
       
        self.get_logger().info(f"Received depth shape {depth_stream[10,20]}")  

    def camera_raw_callback(self,msg):
        '''
        '''
        camera_stream=self._cvbridge.imgmsg_to_cv2(msg)
        self.get_logger().info(f"Received raw imaga shape {camera_stream.shape}") 







def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting webcam streaming  ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    cam_stream = CamStream()
    rclpy.spin(cam_stream)
    # Destroy the node explicitly  
    cam_stream.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()