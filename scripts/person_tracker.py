#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from person_following_robot.msg import Object 
from std_msgs.msg import Int32
from rclpy.logging import LoggingSeverity
from sensor_msgs.msg import Image,PointCloud2
from cv_bridge import CvBridge 
import cv2
import numpy as np



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


class PersonTracker(Node):

    def __init__(self):
        super().__init__('person_tracker')
        ##Subscribers
        #Subscribe to the bounding box values from the person 
        self._subscriber_rec_person_data = self.create_subscription(
            Object,
            'person_following_robot/recognised_person/data',
            self.rec_person_data_callback,
            10)
        self._subscriber_rec_person_data  # prevent unused variable warning
        #Subscribe to the bounding box values from the person recognised
        self._subscriber_camera_image_raw = self.create_subscription(
        Image,
        '/camera/color/image_raw',
        self.camera_image_raw_callback,
        10)
        self._subscriber_camera_image_raw          
        #Subscribe to the depth vales
        self._subscriber_camera_image_depth = self.create_subscription(
        Image,
        '/camera/depth/color/points',
        self.camera_image_raw_callback,
        10)
        self._subscriber_camera_image_depth  


        # Publisher to pubsish person depth
        self.publisher_tracked_person = self.create_publisher(Int32, 'person_following_robot/tracked_person/distance', 10)


        ## Other variables
        self._init_BB=None
        self._cvbridge=CvBridge()
        self._robot_stream=None  
        self._tracker = cv2.TrackerCSRT_create()
        self._first_frame=True
        self._yolo_box_received=False
        self.bounding_box=[10,10,10,10]
        

    def rec_person_data_callback(self, msg):
        ##  Calulate coordintate of top left corner width and height 
        w=int(msg.bounding_box[2]-msg.bounding_box[0])
        l=int(msg.bounding_box[3]-msg.bounding_box[1])
        x=int(msg.bounding_box[0])
        y=int(msg.bounding_box[1])
        self.bounding_box=[x,y,w,l]   
        self._yolo_box_received=True
   



    def camera_image_raw_callback(self,msg):
        '''
        Call back function
        '''
        self._robot_stream=cv2.cvtColor(self._cvbridge.imgmsg_to_cv2(msg) ,cv2.COLOR_BGR2RGB)    
        self.track_object()

    def camera_image_raw_callback(self,msg):
        '''
        Call back function for camera depth
        '''




    def get_depth(self,p1,p2):
        '''Get the distance to the person 
           Removes outliers and get the average distance to 
           the mid point of the bounding box

           Params:
           --------
           p1 : Top left corner of bounding box
           p2 : Bottom right corner of bounding box
        
        '''
        ## TO D0 :
        ## Get the averge outlier removed 
        ## Find a scale invariant method of geting list of 

        mid_x=int((p1[0]+p2[0])/2)
        mid_y=int((p1[1]+p2[1])/2)
        depth=0
        # point_samples
        #  self._position_grid=np.array([np.tile(np.arange(mid_x+10,plane_origin_x-grid_size,-grid_size)[:,None],(1,grid_shape_y)),\
        #                               np.tile(np.arange(plane_origin_y,plane_origin_y-ysize,-grid_size)[:,None].T,(grid_shape_x,1))],dtype=object)




        return 10

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
        if (self._first_frame):
            # initBB = cv2.selectROI('Tracker Frame', cv2.cvtColor(self._robot_stream,cv2.COLOR_RGB2BGR), fromCenter=False)      
            # self._tracker.init(self._robot_stream,initBB)
            self._init_BB=self.bounding_box
            self._tracker.init(self._robot_stream,self.bounding_box)
            cv2.destroyAllWindows()  
            self._first_frame=False
        elif (not self._first_frame) and (self._yolo_box_received):  
            ret, bbox = self._tracker.update(self._robot_stream)
            if ret:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(self._robot_stream, p1, p2, (255,0,0), 2, 1)

                ip1=(int(self._init_BB[0]),int(self._init_BB[1]))
                ip2=(int(self._init_BB[0]+self._init_BB[2]),int(self._init_BB[1]+self._init_BB[3]))
                cv2.rectangle(self._robot_stream, ip1, ip2, (0,255,0), 2, 1)
                cv2.putText(self._robot_stream, "Init BB", ip1, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,2550,0),2)
                distance_to_person=self.get_depth(p1,p2)
            else:
                cv2.putText(self._robot_stream, "Tracking failure detected", (100,80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
                distance_to_person=0
                ##Reinitate tracker
                self._tracker.init(self._robot_stream,self.bounding_box)

        # self.publisher_tracked_person.publish(distance_to_person)
        cv2.imshow("Tracking", self._robot_stream)
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