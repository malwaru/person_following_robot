#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import cv2
from rclpy.logging import LoggingSeverity


###########################################################################
###########################################################################
###########################################################################
## TO DO:
## 1. Create a new msg type to send tracking success
## 2. Use a gaussian filter to cut out the anamolies and get average depth 
## 3. If QR code is not available find the largest box
#############################################################################
#############################################################################
#############################################################################


class WebStream(Node):

    def __init__(self,device=0):
        super().__init__('person_tracker')
  
        self.publisher_webstream = self.create_publisher(Image, '/camera/color/image_raw', 10)
        ## Other variables
        self._cvbridge=CvBridge()
        self._robot_stream=None
        self.camera = cv2.VideoCapture(device)
        self.stream_camera()

        
    def stream_camera(self,device=0):
        '''
        Stream
        '''
        while True: 
            _,stream = self.camera.read()
            # cv2.imshow("Tracking",stream)
            ros_image=self._cvbridge.cv2_to_imgmsg(stream)    
            self.publisher_webstream.publish(ros_image)
            cv2.waitKey(1)
        



def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting webcam streaming  ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    web_stream = WebStream()

    rclpy.spin(web_stream)

    # Destroy the node explicitly  
    web_stream.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()