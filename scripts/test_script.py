#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from rclpy.logging import LoggingSeverity
from person_following_robot.msg import ObjectList, TrackedObject
import cv2
from sort import Sort 
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




        # self.recognised_people=np.empty((0, 5))


            




def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting test script  ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    # test_script = WebStream()
    # test_script=CVTracker()

    test_script=SortTracker()

    rclpy.spin(test_script)

    # Destroy the node explicitly  
    test_script.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()