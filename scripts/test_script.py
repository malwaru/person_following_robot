#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from rclpy.logging import LoggingSeverity
from person_following_robot.msg import ObjectList, TrackedObject
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PoseStamped,PointStamped,Point
from tf2_geometry_msgs import PointStamped
from rclpy.duration import Duration
import time 
from scipy.spatial.transform import Rotation


import cv2
# from sort import Sort 
import numpy as np


class WebStream(Node):

    def __init__(self, device=0):
        super().__init__('web_stream')
        self.publisher_webstream = self.create_publisher(Image,
                                                         '/camera/color/image_raw',
                                                         10)

        # Other variables
        self._cvbridge = CvBridge()
        self._robot_stream = None
        self.camera = cv2.VideoCapture(device)
        self.stream_camera()
        

    def stream_camera(self, device=0):
        '''
        Stream
        '''
        
        _,stream = self.camera.read()
        # cv2.imshow("Tracking",stream)
        ros_image=self._cvbridge.cv2_to_imgmsg(stream)    
        self.publisher_webstream.publish(ros_image)
        cv2.waitKey(1)

class CVTracker(Node):

    def __init__(self, device=0):
        super().__init__('tracker')

        self._subscriber_rec_people_data = self.create_subscription(
                                                ObjectList,
                                                '/person_following_robot/recognised_people/data',
                                                self.rec_people_data_callback,
                                                10)
        self._subscriber_rec_people_data  # prevent unused variable warning
        #Subscribe to the raw image feed
        self._subscriber_camera_image_raw = self.create_subscription(
                                                Image,
                                                '/color/image_raw',
                                                self.camera_image_raw_callback,
                                                10)
        self._subscriber_camera_image_raw 

        self._subscriber_aruco_data = self.create_subscription(
                                        TrackedObject,
                                        '/person_following_robot/aruco_data',
                                        self.aruco_data_callback,
                                        10)
        self._subscriber_aruco_data 
        # Other variables
        self._cvbridge = CvBridge()
        self.robot_stream_colour = None
        self._tracker = cv2.TrackerCSRT_create()
        self._first_frame = True
        self._yolo_box_received = False
        self.aruco_received=False
        self.recognised_people = []

        cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)



    def rec_people_data_callback(self,msg):
        '''
        Subcribe to the messga and output the bouding box in the format 
        coordintate of top left corner width and height     
        '''
        for obj in msg.objects:
            w=int(obj.bounding_box[2]-obj.bounding_box[0])
            l=int(obj.bounding_box[3]-obj.bounding_box[1])
            x=int(obj.bounding_box[0])
            y=int(obj.bounding_box[1])
            self.recognised_people=[[obj.database_id,[x,y,w,l]]]
        self._yolo_box_received=True

        

    def camera_image_raw_callback(self,msg):
        '''
        Subscribe to the image and conver from ROS image messgae 
        and convert to RGB format 
        '''
        self.robot_stream_colour=cv2.cvtColor(self._cvbridge.imgmsg_to_cv2(msg) ,cv2.COLOR_BGR2RGB)
        self.track_object()


    def aruco_data_callback(self,msg):
        '''
        Callback function for the aruco marker data

        Params
        -------
        msg.name    : The name aruco
        msg.id      : Id of the aruco 
        msg.success : Success status of detecting 
        msg.position: position on the frame 
        
        '''
        if msg.success==True:
            self.aruco_received=True
   

    def track_object(self):
        '''
        Tracks the object
        '''
        ## Intialise the tracker only if first RGB frame is received and the aruco marker is received
        if (self._first_frame) and (self.aruco_received): 
            # Check if bounding box inside a leader 
            l_bb_box = cv2.selectROI("Tracking", self.robot_stream_colour)

            self._tracker.init(self.robot_stream_colour,l_bb_box)
            self._first_frame=False


        if (not self._first_frame) and (self._yolo_box_received):  
            ret,bbox = self._tracker.update(self.robot_stream_colour)
            if ret:
                ##Sending the mid point of the bound box to get position function
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(self.robot_stream_colour, p1, p2, (255,0,0), 2, 1)
            else:
                cv2.putText(self.robot_stream_colour, "Tracking failed", (100,80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        cv2.imshow("Tracking",self.robot_stream_colour)
        cv2.waitKey(1)

class SortTracker(Node):

    def __init__(self, device=0):
        super().__init__('tracker')

        self._subscriber_rec_people_data = self.create_subscription(
                                                ObjectList,
                                                '/person_following_robot/recognised_people/data',
                                                self.rec_people_data_callback,
                                                10)
        self._subscriber_rec_people_data  # prevent unused variable warning
        #Subscribe to the raw image feed
        self._subscriber_camera_image_raw = self.create_subscription(
                                                Image,
                                                '/color/image_raw',
                                                self.camera_image_raw_callback,
                                                10)
        self._subscriber_camera_image_raw 

        self._subscriber_aruco_data = self.create_subscription(
                                        TrackedObject,
                                        '/person_following_robot/aruco_data',
                                        self.aruco_data_callback,
                                        10)
        self._subscriber_aruco_data 
        # Other variables
        self._cvbridge = CvBridge()
        self.robot_stream_colour = None
        self._tracker = Sort(max_age=5,min_hits=3,iou_threshold=0.3)
        self.leader_found=False
        self.tracked_idx=-1
        self.aruco_data = {"success": False, "position": [0, 0]}
        self.recognised_people = np.empty((0, 5))
        cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)



    def rec_people_data_callback(self,msg):
        '''
        Subcribe to the messga and output the bouding box in the format 
        coordintate of top left corner width and height     
        '''
    #     dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    # Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    # Returns the a similar array, where the last column is the object ID.
        self.recognised_people=[]
        for obj in msg.objects:
            w=int(obj.bounding_box[2]-obj.bounding_box[0])
            l=int(obj.bounding_box[3]-obj.bounding_box[1])
            x=int(obj.bounding_box[0])
            y=int(obj.bounding_box[1])
            self.recognised_people.append([obj.bounding_box[0],obj.bounding_box[1],obj.bounding_box[2],obj.bounding_box[3],obj.probability])
            # self.recognised_people=np.array([[obj.bounding_box[0],obj.bounding_box[1],obj.bounding_box[2],obj.bounding_box[3],obj.probability]])
        self.recognised_people=np.array(self.recognised_people)
        # self.get_logger().info(f"rec people {self.recognised_people.shape}")
        # self._yolo_box_received=True

        

    def camera_image_raw_callback(self,msg):
        '''
        Subscribe to the image and conver from ROS image messgae 
        and convert to RGB format 
        '''
        self.robot_stream_colour=cv2.cvtColor(self._cvbridge.imgmsg_to_cv2(msg) ,cv2.COLOR_BGR2RGB)
        self.track_object()


    def aruco_data_callback(self,msg):
        '''
        Callback function for the aruco marker data

        Params
        -------
        msg.name    : The name aruco
        msg.id      : Id of the aruco 
        msg.success : Success status of detecting 
        msg.position: position on the frame 
        
        '''
        if msg.success==True:
            self.aruco_received=True

        self.aruco_data["success"]=msg.success
        self.aruco_data["position"]=msg.position
   

    def check_leader(self,tracked_people):
        '''
        Check if the people in the bounding box is also marked with an 
        aruco marker a leader

        Returns
        --------
        If Leader was found and the bounding box
        '''    
        bbox=[]    
        #Check if tracking was successfull
        if len(tracked_people)>0:            
            for person in tracked_people:
                x1=int(person[0])
                y1=int(person[1])
                x2=int(person[2])
                y2=int(person[3])
                ## If the the leader unknown 
                if self.tracked_idx == -1: 
                    # aruco marker is found
                    if self.aruco_data["success"]==True:
                        aruco_center=[int(self.aruco_data["position"][0]),int(self.aruco_data["position"][1])]
                        if (aruco_center[0]>=x1) and (aruco_center[0]<=x2) and (aruco_center[1]>=y1) and (aruco_center[1]<=y2):
                            self.leader_found=True
                            bbox=((x1,y1),(x2,y2))   
                            self.tracked_idx=person[4]
                            self.get_logger().info(f"Debug : Case 1")
                ## If tracked index is available in the current index
                else:
                    if person[4]==self.tracked_idx : 
                        bbox=[(x1,y1),(x2,y2)]
                        self.get_logger().info(f"Debug : Case 2")
                        break
                    # else:
                    #     self.leader_found=False
                    #     self.tracked_idx=-1
                    #     self.get_logger().info(f"Debug : Case 4")
            
             
        else:
            self.leader_found=False
            self.get_logger().info(f"Debug : Case 3 {tracked_people}")
            self.tracked_idx=-1

        return bbox

    def check_leader_2(self,tracked_people):
        '''
        Check if the people in the bounding box is also marked with an 
        aruco marker a leader

        Returns
        --------
        If Leader was found and the bounding box
        '''    
        bbox=[]    
        #Check if tracking was successfull
        if len(tracked_people)>0:
            
            # self.get_logger().info(f"Debug : Case 0 id : {self.tracked_idx} lead : {self.leader_found} \n tracked pp: {tracked_people}")
            ## If the the leader unknown aruco is available
            if (not self.leader_found) and (self.aruco_data["success"]==True): 
                aruco_center=[int(self.aruco_data["position"][0]),int(self.aruco_data["position"][1])]
                self.get_logger().info(f"Debug : Case 1")
                # aruco marker is found
                for person in tracked_people:
                    x1=int(person[0])
                    y1=int(person[1])
                    x2=int(person[2])
                    y2=int(person[3])                
                    if (aruco_center[0]>=x1) and (aruco_center[0]<=x2) and (aruco_center[1]>=y1) and (aruco_center[1]<=y2):
                        self.leader_found=True
                        bbox=((x1,y1),(x2,y2))   
                        self.tracked_idx=person[4]
                        self.get_logger().info(f"Debug : Case 2")
            ## If tracked index is available in the current index
            else:
                ##This make sure if due to occulusion the idx was changed the person can still be tracked 
                self.leader_found=False
                for person in tracked_people:
                    x1=int(person[0])
                    y1=int(person[1])
                    x2=int(person[2])
                    y2=int(person[3])
                    if person[4]==self.tracked_idx :
                        self.leader_found=True 
                        bbox=[(x1,y1),(x2,y2)]
                        self.get_logger().info(f"Debug : Case 3")   
                # if not self.leader_found:
                #     self.leader_found=

        else:
            self.leader_found=False
            self.get_logger().info(f"Debug : Case 4 {tracked_people}")
            self.tracked_idx=-1

        return bbox

    def track_object(self):
        '''
        Tracks the object
        '''
    
        tracked_people=self._tracker.update(self.recognised_people)
        track_bbox=self.check_leader_2(tracked_people)

        if  self.leader_found and len(track_bbox)>0:        
            self.get_logger().info(f"\n Got rectangle p1: {track_bbox[0]}, p2: {track_bbox[1]}")
            cv2.rectangle(self.robot_stream_colour, track_bbox[0], track_bbox[1], (255,0,0), 2, 1)
        else:
            cv2.putText(self.robot_stream_colour, "Leader not found", (100,80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            # self.get_logger().info(f"\n Debug Case 5 : tracker out {len(tracked_people)} lead out :{track_bbox}")
        cv2.imshow("Tracking",self.robot_stream_colour)
        cv2.waitKey(1)

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
 
class ParameterCheck(Node):
    def __init__(self) -> None:
        super().__init__('tracker')        
        
        self._subscriber_rec_people_data = self.create_subscription(
                                            Image,
                                            '/aligned_depth_to_color/image_raw',
                                            self.camera_image_depth_callback,
                                            10)


        self._cvbridge=CvBridge()

    def camera_image_depth_callback(self,msg):
        '''
        M
        '''
        robot_stream_depth = self._cvbridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") 
        ## The scale of depth pixels is 0.001|  16bit depth, one unit is 1 mm, 
        self.robot_stream_depth = np.array(robot_stream_depth, dtype=np.uint16)*0.001

        self.get_logger().info(f"Depth shape {self.robot_stream_depth.shape}")


class TfTransform(Node):
    def __init__(self):
        super().__init__('tfseer')        
   
        self._subscriber_camera_image_raw = self.create_subscription(
                                        Image,
                                        '/camera/color/image_raw',
                                        self.camera_image_raw_callback,
                                        10)
        self.frame_source = self.declare_parameter(
          'source_frame', 'camera_color_optical_frame').get_parameter_value().string_value
        self.frame_target = self.declare_parameter(
          'target_frame', 'base_link').get_parameter_value().string_value
        # self.frame_source='camera_color_optical_frame'
        # self.frame_target='base_link'

        self.tracked_person_position= PoseStamped()
        self.tracked_person_point= PointStamped()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)
        # self.check_transform()
        # self.timer = self.create_timer(0.2, self.check_transform)
        self.tracked_person_position.header.frame_id=self.frame_source
        self.tracked_person_position.pose.orientation.x = 0.0
        self.tracked_person_position.pose.orientation.y = 0.0
        self.tracked_person_position.pose.orientation.z = 0.0
        self.tracked_person_position.pose.orientation.w = 1.0

        self.tracked_person_point.header.frame_id = 'camera_depth_optical_frame'




    def coordinate_transform(self,t_x,t_y,t_z,r_x,r_y,r_z,r_w,position):
        '''
        '''
        self.tracked_person_point.point.x = float(np.random.randint(0,30))
        self.tracked_person_point.point.y = float(np.random.randint(0,30))
        self.tracked_person_point.point.z = float(np.random.randint(0,30)) 

        rot_x=tranformer.transform.rotation.x
        rot_y=tranformer.transform.rotation.y
        rot_z=tranformer.transform.rotation.z
        rot_w=tranformer.transform.rotation.w
        tran_x=tranformer.transform.translation.x
        tran_y=tranformer.transform.translation.x
        tran_z=tranformer.transform.translation.z


    def coordinate_transform_2(self,transformer,position):
        '''
        '''
      
        rot_x=tranformer.transform.rotation.x
        rot_y=tranformer.transform.rotation.y
        rot_z=tranformer.transform.rotation.z
        rot_w=tranformer.transform.rotation.w
        tran_x=tranformer.transform.translation.x
        tran_y=tranformer.transform.translation.x
        tran_z=tranformer.transform.translation.z


    def camera_image_raw_callback(self,msg):
        self.tracked_person_position.header.stamp=self.get_clock().now().to_msg()
        self.tracked_person_position.pose.position.x = float(np.random.randint(0,30))
        self.tracked_person_position.pose.position.y = float(np.random.randint(0,30))
        self.tracked_person_position.pose.position.z = float(np.random.randint(0,30))  


        self.tracked_person_point.point.x = float(np.random.randint(0,30))
        self.tracked_person_point.point.y = float(np.random.randint(0,30))
        self.tracked_person_point.point.z = float(np.random.randint(0,30))    

        

        # if self.tf_buffer.can_transform('camera_depth_optical_frame','base_link',rclpy.time.Time()):
        #     try:
        #         tranformer=self.tf_buffer.lookup_transform(source_frame=self.frame_source,target_frame=self.frame_target,time=rclpy.time.Time())
        #         self.tracked_person_position.header.frame_id=self.frame_source

        #         # self.tracked_person_position.header.stamp=self.get_clock().now().to_msg()
        #         self.tracked_person_position.pose.position.x = float(np.random.randint(0,30))
        #         self.tracked_person_position.pose.position.y = float(np.random.randint(0,30))
        #         self.tracked_person_position.pose.position.z = float(np.random.randint(0,30))  
        #         self.tracked_person_position=self.tf_buffer.transform(self.tracked_person_position,'base_link')
        #         self.get_logger().info(f"Found Transform")

        #     except TransformException as ex:
        #         self.get_logger().info(f"Could to find tf transform {ex}")

        # else:
        #     self.get_logger().info(f"can_transform failed")

        try:
            tranformer=self.tf_buffer.lookup_transform(source_frame=self.frame_source,target_frame=self.frame_target,time=rclpy.time.Time())
            rot_x=tranformer.transform.rotation.x
            rot_y=tranformer.transform.rotation.y
            rot_z=tranformer.transform.rotation.z
            rot_w=tranformer.transform.rotation.w
            tran_x=tranformer.transform.translation.x
            tran_y=tranformer.transform.translation.x
            tran_z=tranformer.transform.translation.z
            self.coordinate_transform(tran_x,tran_x)
            rot_quat=Rotation.from_quat('xyzw', [rot_x,rot_y, rot_z,rot_w], degrees=False)
            rot_euler=rot_quat.as_euler()
            # self.tracked_person_point = tf2_geometry_msgs.do_transform_point(self.tracked_person_point, transform)

            self.get_logger().info(f'Running tranform')

        except TransformException as ex:
            self.get_logger().info(f"Could to find tf transform {ex}")


            

    def check_transform(self):



        self.tracked_person_position.header.frame_id = self.frame_target
        self.tracked_person_position.point.x = float(np.random.randint(0,30))
        self.tracked_person_position.point.y = float(np.random.randint(0,30))
        self.tracked_person_position.point.z = float(np.random.randint(0,30)) 
        duration = Duration(seconds=1.1,nanoseconds=0)


        # try:
        #     tranformer=self.tf_buffer.lookup_transform(source_frame=self.frame_source,target_frame=self.frame_target,time=rclpy.time.Time())
        #     # self.tracked_person_pose=self.tf_buffer.transform(self.tracked_person_position,'base_link')
        #     self.get_logger().info(f'Running tranform')

        # except TransformException as ex:
        #     self.get_logger().info(f"Could to find tf transform {ex}")


def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting test script  ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    # test_script = WebStream()
    # test_script=CVTracker()
    test_script=TfTransform()
    rclpy.spin(test_script)
    # Destroy the node explicitly  
    test_script.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()