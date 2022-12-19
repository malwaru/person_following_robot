#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from geometry_msgs.msg import PoseStamped,PointStamped,Point
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from rclpy.logging import LoggingSeverity
from person_following_robot.msg import ObjectList, TrackedObject
import cv2
from sort import Sort 
import numpy as np
import pyrealsense2 as rs
import copy
from scipy import stats as st
import matplotlib.pyplot as plt




########################
##  To Do
########################
## 1. Load the sort algorithms params from a config file
## 2. Get real depth camera intriscis/ load from yaml
##      a. check the source path 


class SortTracker(Node):

    def __init__(self, device=0): 
        super().__init__('person_tracker')
    

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


        self.publisher_tracked_person = self.create_publisher(
                                        PointStamped,
                                        'tracked_person/position', 
                                        10)
        # Other variables
        self._cvbridge = CvBridge()
        self.robot_stream_colour = None

        ##Tracker related
        self._tracker = Sort(max_age=1,min_hits=3,iou_threshold=0.3)
        self.leader_found=False
        self.tracked_idx=-1
        self.aruco_data = {"success": False, "position": [0, 0]}
        self.recognised_people = np.empty((0, 5))
        cv2.namedWindow("Tracking", cv2.WINDOW_AUTOSIZE)
        self.tracked_person_position = PointStamped()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)



        ## Pyrealsense instrics of the depth camera 
        self.depth_intrinsic = rs.intrinsics()
        self.depth_intrinsic.width = 640
        self.depth_intrinsic.height = 480
        self.depth_intrinsic.ppx = 322.043121337891
        self.depth_intrinsic.ppy = 238.831329345703
        self.depth_intrinsic.fx = 393.181854248047
        self.depth_intrinsic.fy = 393.181854248047
        self.depth_intrinsic.model = rs.distortion.brown_conrady
        self.depth_intrinsic.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]



        ## Plot variabls
        self.figure1=plt.figure()
        self.ax1=self.figure1.add_subplot(111)
        self.ax1.set(title="Depth_distribution",xlabel="Points",ylabel="Depth(m)")





    def rec_people_data_callback(self,msg):
        '''
        Subcribe to the messga and output the bouding box in the format 
        coordintate of top left corner width and height     
        '''
        self.recognised_people=[]
        for obj in msg.objects:        
            self.recognised_people.append([obj.bounding_box[0],obj.bounding_box[1],obj.bounding_box[2],obj.bounding_box[3],obj.probability])
        self.recognised_people=np.array(self.recognised_people)
             

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
        The size of the image is (480, 640)

        '''
        robot_stream_depth = self._cvbridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") 
        ## The scale of depth pixels is 0.001|  16bit depth, one unit is 1 mm, 
        self.robot_stream_depth = np.array(robot_stream_depth, dtype=np.uint16)*0.001


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
            ## If the leader is unknown and aruco is available
            if (not self.leader_found) and (self.aruco_data["success"]==True): 
                aruco_center=[int(self.aruco_data["position"][0]),int(self.aruco_data["position"][1])]
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
            ## If tracked index is available in the current index
            else:
                ##This make sure if occulusion changed idx ,the person can still be tracked 
                self.leader_found=False
                for person in tracked_people:
                    x1=int(person[0])
                    y1=int(person[1])
                    x2=int(person[2])
                    y2=int(person[3])
                    if person[4]==self.tracked_idx :
                        self.leader_found=True 
                        bbox=[(x1,y1),(x2,y2)]       

        else:
            self.leader_found=False
            self.tracked_idx=-1

        return bbox

    def get_position_filtered_mean(self,p1,p2):
        '''Removes outliers and get the mean distance

            This method is intented to be used in the case where the 
            middle point of the bounding box does not fall on the 
            person 
           

           Params:
           --------
           p1 : Top left corner of bounding box
           p2 : Bottom right corner of bounding box

           Returns:
           -------
           position [x,y,z] the 3D coordinates of pixels p1,p2 the 3D points in camera frame 
           
        
        '''
        ## TO D0 :
        ## 1. Try the mean and median depths

        depth_list=[]
        w= (self.robot_stream_depth.shape)[0]
        l= (self.robot_stream_depth.shape)[1]
        x1=np.clip(p1[0],0,w-1)
        x2=np.clip(p2[0],0,w-1)
        y1=np.clip(p1[1],0,l-1)
        y2=np.clip(p2[1],0,l-1)
        depth_list=[self.robot_stream_depth[x,y] for x in range(x1,x2) for y in range(y1,y2)]
        depth_list=np.asarray(depth_list)
        ##
        self.ax1.cla()
        self.ax1.scatter(range(len(depth_list)),depth_list)
        plt.pause(0.001)


        depth_copy_list=copy.deepcopy(depth_list)
        new_depth_list=self.remove_outliers(depth_copy_list)
        mean_depth=np.mean(new_depth_list)      
        depth=mean_depth
        depth_point=self.closest_point(depth,(x1,x2,y1,y2))
        position=[depth_point[0],depth_point[1],depth]
        # position=rs.rs2_deproject_pixel_to_point([float(mid_x),float(mid_y)],float(mid_z))


        return position

    def closest_point(self,depth,corners):
        '''
        Find the index point that the depth value is closest to
        '''
        min_dis=np.Inf
        x0=corners[0]
        y0=corners[2]
        for x in range(corners[0],corners[1]):
            for y in range(corners[2],corners[3]):
                current_depth=-self.robot_stream_depth[x,y]
                if abs(depth-current_depth)<min_dis:
                    x0=x
                    y0=y
                    min_dis=abs(depth-current_depth)




        return (x0,y0)

    def remove_outliers(self,data, pp1 = 0.01, pp2 = 0.001) -> np.array:
        '''
        Detect and remove outliers based on Chebychev Theorem
        
        Returns
        ---------
        outliers_detected:  Number of outliers detected
        final_data_indices: Indices for filtered data        '''

        
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


    def get_position_middle(self,p1,p2):
        '''
        Output the distance of the mid point of the bounding box    

        Params:
        --------
        p1 : Top left corner of bounding box
        p2 : Bottom right corner of bounding box

        Returns:
        -------
        position [x,y,z] the 3D coordinates of pixels p1,p2 the 3D points in camera frame 
        
        
        '''        
        w= (self.robot_stream_depth.shape)[0]
        l= (self.robot_stream_depth.shape)[1]
        x1=np.clip(p1[0],0,w-1)
        x2=np.clip(p2[0],0,w-1)
        y1=np.clip(p1[1],0,l-1)
        y2=np.clip(p2[1],0,l-1)
        x= int((x1+x2)/2)
        y= int((y1+y2)/2)       
        depth=self.robot_stream_depth[y,x]             
        pixel_point=[float(x),float(y)]
        ##Getting position in depth camera frame
        position=rs.rs2_deproject_pixel_to_point(self.depth_intrinsic,pixel_point,depth)
 


        return position


    def track_object(self):
        '''
        Tracks the object
        '''
    
        tracked_people=self._tracker.update(self.recognised_people)
        track_bbox=self.check_leader(tracked_people)

        if  self.leader_found and len(track_bbox)>0:        
            cv2.rectangle(self.robot_stream_colour, track_bbox[0], track_bbox[1], (255,0,0), 2, 1)
            ##Getting position in depth camera frame
            pos=self.get_position_middle(track_bbox[0], track_bbox[1])
            print_pos="x "+str(np.round(pos[2],3))+" m  y:"+str(pos[0])
            cv2.putText(self.robot_stream_colour, print_pos, (track_bbox[0][0]+10,track_bbox[0][1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,2550,0),2)

            
            ##Pose in camera depth frame
            # self.tracked_person_pose.header.stamp = self.get_clock().now().to_msg()
            self.tracked_person_position.header.frame_id = 'camera_depth_optical_frame'
            #In the camera Frame Z in the X direction in robot coordinate system 
            #Here we take that into account
            self.tracked_person_position.point.x = pos[2]
            self.tracked_person_position.point.y = pos[0]
            self.tracked_person_position.point.z = pos[1]
        
            ##Check if the transformation is available 
            
            if self.tf_buffer.can_transform('camera_depth_optical_frame','base_link',self.get_clock().now()):
                try:
                    #Transform the pose to base_link frame # Source https://w3.cs.jmu.edu/spragunr/CS354_S19/lectures/tf/tf2_demo.py
                    self.tracked_person_pose=self.tf_buffer.transform(self.tracked_person_position,'base_link')
                except TransformException as ex:
                    self.get_logger().info(f"Could to find tf transform {ex}")
                self.publisher_tracked_person.publish(self.tracked_person_position)
    
            
        else:
            cv2.putText(self.robot_stream_colour, "Leader not found", (100,80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        cv2.imshow("Tracking",self.robot_stream_colour)
        cv2.waitKey(1) 



def main(args=None):
    rclpy.logging._root_logger.log(
        'Starting person tracker  ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)

    person_tracker=SortTracker()

    rclpy.spin(person_tracker)

    # Destroy the node explicitly  
    person_tracker.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()