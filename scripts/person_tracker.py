#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from person_following_robot.msg import ObjectList, TrackedObject
from rclpy.logging import LoggingSeverity
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pyrealsense2 as rs

###########################################################################
###########################################################################
###########################################################################
## TO DO:
## 1. Use the XYZ to find world coordinates
## 2. Use a gaussian filter to cut out the anamolies and get average depth 
##
#############################################################################
#############################################################################
#############################################################################


class PersonTracker(Node):

    def __init__(self):
        super().__init__('person_tracker')
        ##Subscribers
        #Subscribe to the bounding box values from the person 
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

        self._subscriber_camera_image_depth = self.create_subscription(
                                                    Image,
                                                    '/aligned_depth_to_color/image_raw',
                                                    self.camera_image_depth_callback,
                                                    10)
        self._subscriber_camera_image_depth  

        self._subscriber_aruco_data = self.create_subscription(
                                                TrackedObject,
                                                '/person_following_robot/aruco_data',
                                                self.aruco_data_callback,
                                                10)
        self._subscriber_aruco_data 


        # Publisher to pubsish person depth
        self.publisher_tracked_person_data = self.create_publisher(
                                                TrackedObject,
                                                'tracked_person/data', 
                                                10)
        self.publisher_tracked_person_image = self.create_publisher(
                                                Image,
                                                'tracked_person/image', 
                                                10)

        ## Misc variables
        self._init_BB = None
        self._cvbridge = CvBridge()
        self.robot_stream_colour = None
        self.robot_stream_depth = None
        self.depth_point_stream_robot = None
        self.aruco_data = {"success": False, "position": [0, 0]}
        self._tracker = cv2.TrackerCSRT_create()
        self._first_frame = True
        self._yolo_box_received = False
        self.recognised_people = []
        cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
        self.image_size = (480, 640)
        # self.video_out = cv2.VideoWriter('./output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 10.0,(640,480))
        self.stop_record = False
   
            

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

    def camera_image_depth_callback(self,msg):
        '''
        Call back function for camera depth and decode
        using passthrough
        '''
        self.robot_stream_depth = self._cvbridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") 

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
        self.aruco_data["success"]=msg.success
        self.aruco_data["position"]=msg.position

    def check_leader(self):
        '''
        Check if the people in the bounding box is also marked with an 
        aruco marker a leader

        Returns
        --------
        If Leader was found and the bounding box
        '''
        leader_status=False
        bounding_box=[]  
        if self.aruco_data["success"]==True:
            aruco_center=[int(self.aruco_data["position"][0]),int(self.aruco_data["position"][1])]
            for person in self.recognised_people:
                x=person[1][0]
                y=person[1][1]
                w=person[1][2]
                l=person[1][3]
                if (aruco_center[0]>=x) and (aruco_center[0]<=x+w) and (aruco_center[1]>=y) and (aruco_center[1]<=y+l):
                    bounding_box=person[1]
                    leader_status=True
        return leader_status,bounding_box


    def get_position(self,p1,p2):
        '''Get the distance to the person 
           Removes outliers and get the average distance to 
           the mid point of the bounding box

           Params:
           --------
           p1 : Top left corner of bounding box
           p2 : Bottom right corner of bounding box

           Returns:
           -------
           position [x,y,z] the 3D coordinates of pixels p1,p2 the 3D points in camera frame 
           
        
        '''
        ## TO D0 :
        ## Get the outlier removed when finding the average mid point
        ## Find a scale invariant method of geting list of point of the bounding box mid  
        #   
        
        mid_x=int((p1[0]+p2[0])/2)
        mid_y=int((p1[1]+p2[1])/2)
        position=[0.,0.,0.]            
        mid_z=self.robot_stream_depth[mid_y-1,mid_x-1]            
        position=[mid_x/1000,mid_y/1000,mid_z/1000]

            # position=rs.rs2_deproject_pixel_to_point([float(mid_x),float(mid_y)],float(mid_z))
            # self.get_logger().info(f"Received world {position}")      


        return position

    def get_position_2(self,p1,p2):
        '''Get the distance to the person 
           Removes outliers and get the average distance to 
           the mid point of the bounding box

           Params:
           --------
           p1 : Top left corner of bounding box
           p2 : Bottom right corner of bounding box

           Returns:
           -------
           position [x,y,z] the 3D coordinates of pixels p1,p2 the 3D points in camera frame 
           
        
        '''
        ## TO D0 :
        ## Get point with min distance in the bounding box        
        min_dis=0.
        position=[0.,0.,0.]
        # for x in range(p1[0],p2[0]):
        #     for y in range(p1[1],p2[1]):
        #         current_dis=self.robot_stream_depth[y-2,x-2]  
        #         if current_dis<=min_dis:
        #             min_dis=current_dis
        #             position[0]=float(x)
        #             position[1]=float(y)
        #             position[2]=current_dis

        depth_list=[self.robot_stream_depth[y-1,x-1] for x in range(p1[0],p2[0]) for y in range(p1[1],p2[1])]
        self.get_logger().info(f"Filtered depth {depth_list[3]}")
        # new_depth_list=np.asarray(self.remove_outliers(depth_list))
        # average=np.average(new_depth_list)
        # self.get_logger().info(f"Filtered depth {average}")


        return position

    def remove_outliers(data, pp1 = 0.01, pp2 = 0.001) -> np.array:
        '''
        Detect outliers based on Chebychev Theorem
        
        Returns
        ---------
        outliers_detected:  Number of outliers detected
        final_data_indices: Indices for filtered data
        '''
        
        mu1=np.mean(data)
        sigma1=np.var(data)
        k = 1 / np.sqrt(pp1)
        odv1u = mu1 + k * sigma1
        odv1l = mu1 - k * sigma1        
        new_data = data[np.where(data <= odv1u)[0]]        
        new_data = new_data[np.where(new_data >= odv1l)[0]]        
        mu2=np.mean(new_data)
        sigma2=np.var(new_data)
        k = 1 / np.sqrt(pp2)
        odv2u = mu2 + k * sigma2
        odv2l = mu2 - k * sigma2
        final_data = new_data[np.where(new_data <= odv2u)[0]]        
        final_data = new_data[np.where(final_data >= odv2l)[0]]
        
        return final_data


    def track_object(self):
        '''
        Tracks the object
        '''
        ## Intialise the tracker only if first RGB frame is received and the aruco marker is received
        if (self._first_frame) and (self._yolo_box_received): 
            # Check if bounding box inside a leader 
            leader_sucess,l_bb_box=self.check_leader()
            if leader_sucess:
                self._init_BB=l_bb_box    
                self._tracker.init(self.robot_stream_colour,l_bb_box)
                self._first_frame=False


        if (not self._first_frame) and (self._yolo_box_received):  
            ret,bbox = self._tracker.update(self.robot_stream_colour)
            if ret:
                ##Sending the mid point of the bound box to get position function
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(self.robot_stream_colour, p1, p2, (255,0,0), 2, 1)
                position=self.get_position(p1,p2)         
                _=self.get_position_2(p1,p2)         
                print_pos="Depth: "+str(position[2])+"m"
                cv2.putText(self.robot_stream_colour, print_pos, (p1[0]+10,p1[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,2550,0),2)
                track_person_data=TrackedObject(name="Person",id=1,success=True,position=position)
                self.publisher_tracked_person_data.publish(track_person_data)
 
                
            else:
                cv2.putText(self.robot_stream_colour, "Tracking failed", (100,80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                track_person_data=TrackedObject(name="Person",id=0,success=False,position=[])
                self.publisher_tracked_person_data.publish(track_person_data)
                ##Reinitate tracker
                leader_sucess,l_bb_box=self.check_leader()
                if leader_sucess:
                    self._init_BB=l_bb_box
                    self._tracker.init(self.robot_stream_colour,l_bb_box)


        cv2.imshow("Tracking",self.robot_stream_colour)
        self.publisher_tracked_person_image.publish(self._cvbridge.cv2_to_imgmsg(self.robot_stream_colour))
        cv2.waitKey(1)

def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting person tracking ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)
    person_tracker = PersonTracker()
    rclpy.spin(person_tracker)
    # Destroy the node explicitly  
    person_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()