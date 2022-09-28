#!/usr/bin/env python3
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from rclpy.logging import LoggingSeverity
from person_following_robot.msg import TrackedObject

###########################################################################
###########################################################################
###########################################################################
## TO DO:
## 1. Search for markers only when the tracking is missing
## 2. Track multiple aruco
## 3. Load topic names from params 
#############################################################################
#############################################################################
#############################################################################


class ArucoDetector(Node):
    def __init__(self) -> None:
        super().__init__('aruco_detector')
        self._subscriber_camera_image_raw = self.create_subscription(
                                                Image,
                                                '/color/image_raw',
                                                self.camera_image_raw_callback,
                                                10)
        self._subscriber_camera_image_raw  # prevent unused variable warning

        # Publisher to pubsish person depth
        self.publisher_aruco_data = self.create_publisher(
                                                TrackedObject,
                                                'aruco_data', 
                                                10)



        self._robot_stream_colour = None
        self._cvbridge=CvBridge()
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()



    def camera_image_raw_callback(self,msg):
        '''
        Receive Image message
        
        '''

        self._robot_stream_colour=cv2.cvtColor(self._cvbridge.imgmsg_to_cv2(msg) ,cv2.COLOR_BGR2RGB)   
        self.detect_marker()

    def detect_marker(self):
        '''
        Returns
        --------
        Windows
        '''

        stream=self._robot_stream_colour
        (corners, ids, rejected) = cv2.aruco.detectMarkers(stream, self.arucoDict,parameters=self.arucoParams)
        topLeft=[20,50]
        topRight=[10,10]
        # cv2.namedWindow("Image",cv2.WINDOW_FREERATIO)

        if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
            # loop over the detected ArUCo corners
            for (markerCorner, markerID) in zip(corners, ids):
                # extract the marker corners (which are always returned in
                # top-left, top-right, bottom-right, and bottom-left order)
                corners = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = corners
                # convert each of the (x, y)-coordinate pairs to integers
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                topLeft = (int(topLeft[0]), int(topLeft[1]))


            # draw the bounding box of the ArUCo detection
                cv2.line(stream, topLeft, topRight, (0, 255, 0), 2)
                cv2.line(stream, topRight, bottomRight, (0, 255, 0), 2)
                cv2.line(stream, bottomRight, bottomLeft, (0, 255, 0), 2)
                cv2.line(stream, bottomLeft, topLeft, (0, 255, 0), 2)
                # compute and draw the center (x, y)-coordinates of the ArUco
                # marker
                cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                cv2.circle(stream, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the stream
                cv2.putText(stream, str(markerID),
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
                marker_data=TrackedObject(name="Aruco",id=int(markerID),success=True,position=[cX,cY])
                self.publisher_aruco_data.publish(marker_data)

                # print(f"[INFO] ArUco marker ID: {markerID}")
                # show the output image
        else:
                cv2.putText(stream,"No marker",
                    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
                marker_data=TrackedObject(name="Aruco",id=0,success=False,position=[])
                self.publisher_aruco_data.publish(marker_data)
        # cv2.imshow("Image", stream)
        # cv2.waitKey(1)


def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting aruco marker detection...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    rclpy.spin(aruco_detector)
    # Destroy the node explicitly  
    aruco_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
 