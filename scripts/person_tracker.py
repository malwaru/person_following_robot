#!/usr/bin/env python3
#ROS related
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
from rclpy.logging import LoggingSeverity
from geometry_msgs.msg import PoseStamped,PointStamped
from person_following_robot.msg import TrackedObject
import pyrealsense2 as rs
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException
from geometry_msgs.msg import PointStamped
from scipy.spatial.transform import Rotation



#Bytetracker related 
import os
import os.path as osp
import cv2
import numpy as np
import torch
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess,get_path
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer



class ByteParams():
    '''
    The parametes needed for the ByteTracker and  YoloX detector
    This replaces the original argparse 
    '''

    def __init__(self,exp_file="",
                model_file="",
                device="gpu",
                fp16=False,
                fuse=False) -> None:
        ##Detector(YoloX) parameters
        self.name=None
        self.exp_file=exp_file
        self.ckpt=model_file
        self.camid=0
        self.device=device
        self.conf=None #Float test confidence
        self.nms=None # Float test nms threshold
        self.tsize=None # int test image size
        self.fps=30 # int frame rate
        self.fp16=fp16 # Bool action="store_true", help="Adopting mix precision evaluating
        self.fuse=fuse # Bool  action="store_true",help="Fuse conv and bn for testing
        self.trt=False # Bool  action="store_true", help="Using TensorRT model for testing
        ##Tracking parameters
        self.track_thresh=0.5 # help="tracking confidence threshold
        self.track_buffer = 30 # the frames for keep lost tracks
        self.match_thresh = 0.8 # matching threshold for tracking
        self.aspect_ratio_thresh=1.6 # threshold for filtering out boxes of which aspect ratio are above the given value
        self.min_box_area=10 # filter out tiny boxes
        self.mot20=False # test mot20.

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):

        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
        return outputs, img_info




class TrackerByte(Node):
    def __init__(self,predictor,exp,args):
        super().__init__('byte_tracker')
        self._subscriber_camera_image_raw = self.create_subscription(
                                                Image,
                                                '/camera/color/image_raw',
                                                self.camera_image_raw_callback,
                                                10)
        self._subscriber_camera_image_raw 
        
        self._subscriber_camera_image_depth = self.create_subscription(
                                                    Image,
                                                    '/camera/aligned_depth_to_color/image_raw',
                                                    self.camera_image_depth_callback,
                                                    10)
        self._subscriber_camera_image_depth  

        self._subscriber_aruco_data = self.create_subscription(
                                        TrackedObject,
                                        '/person_following_robot/aruco_data',
                                        self.aruco_data_callback,
                                        10)
        self._subscriber_aruco_data 


        self.publisher_tracked_person_position = self.create_publisher(
                                        PointStamped,
                                        'tracked_person/position', 
                                        10)

        self.publisher_tracked_person_image_raw = self.create_publisher(
                                        Image,
                                        'tracked_person/image_raw', 
                                        10)

        ## Pyrealsense depth instrics of the depth camera taken running the command 'rs-enumerate-devices -c'in terminal
        self.depth_intrinsic = rs.intrinsics()
        self.depth_intrinsic.width = 640
        self.depth_intrinsic.height = 480
        self.depth_intrinsic.ppx = 317.862670898438
        self.depth_intrinsic.ppy =239.181976318359
        self.depth_intrinsic.fx = 384.224700927734
        self.depth_intrinsic.fy = 384.224700927734
        self.depth_intrinsic.model = rs.distortion.brown_conrady
        self.depth_intrinsic.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]

        #Tracking leader
        self.tracked_idx=-1
        self.aruco_data = {"success": False, "position": [0, 0]}
        ## Transforming coodinates
        self.tracked_person_position = PointStamped()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer,self)
        self.frame_source = self.declare_parameter(
          'source_frame', 'camera_color_optical_frame').get_parameter_value().string_value
        self.frame_target = self.declare_parameter(
          'target_frame', 'base_link').get_parameter_value().string_value    
        self.transform_acquired_base_camera=False


        self.record_video=False
        #Testing related
        _sub=str(np.random.randint(0,10000))
        if self.record_video:
            self.video_recorder = cv2.VideoWriter('/home/romatris/Workspace/Misc/TestVideos/A/a1.3.4-a1.3.6_trial_2_'+_sub+'.mp4', 
                            cv2.VideoWriter_fourcc(*'MJPG'),10, (640,480))
            ###
        self.cvbridge = CvBridge()
        self.robot_stream_colour = None 
        self.predictor=predictor
        self.args=args
        self.exp=exp
        self.tracker = BYTETracker(self.args, frame_rate=30)
        self.timer = Timer()
        self.frame_id = 0
        self.results = []

    def camera_image_raw_callback(self,msg):
        '''
        Raw image of camera
        '''
        self.robot_stream_colour=cv2.cvtColor(self.cvbridge.imgmsg_to_cv2(msg) ,cv2.COLOR_BGR2RGB)
        self.imageflow_demo()


    def camera_image_depth_callback(self,msg):

        '''
        Call back function for camera depth and decode
        using passthrough
        The size of the image is (480, 848)
        Source : 
        https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
        https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20

        https://github.com/IntelRealSense/realsense-ros/issues/1870

        '''
        robot_stream_depth = self.cvbridge.imgmsg_to_cv2(msg, desired_encoding="16UC1") 
        ## The scale of depth pixels is 0.001|  16bit depth, one unit is 1 mm | taken from data sheet 
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
       
        self.aruco_data["success"]=msg.success
        self.aruco_data["position"]=msg.position 

    def check_leader(self,tracked_bboxs,ids):
        '''
        Check if the people in the bounding box is also marked with an 
        aruco marker a leader

        Parameters
        ----------
        tracked_bboxs : (x1,y1,w,h) 

        Returns
        --------
        Bbox corners values   
        '''
        bbox=[]  
        lead_idx=None
        #Check if the tracked leader id is still avaialble 
        for index,id in enumerate(ids):
            if id==self.tracked_idx:
                lead_idx=index
        ## If leader is unknown and marker is available look for leader
        if (lead_idx is None) and (self.aruco_data["success"]==True):
            aruco_center=[int(self.aruco_data["position"][0]),int(self.aruco_data["position"][1])]                
            for index,id in enumerate(ids):
                x1=(tracked_bboxs[index][0])
                y1=(tracked_bboxs[index][1])
                w=(tracked_bboxs[index][2])
                h=(tracked_bboxs[index][3]) 
                if (aruco_center[0]>=x1) and (aruco_center[0]<=(x1+w)) and (aruco_center[1]>=y1) and (aruco_center[1]<=(y1+h)):
                    bbox=[[int(x1),int(y1)],[int(x1+w),int(y1+h)]]   
                    self.tracked_idx=id
                    self.get_logger().info(f"Found leader with Id: {id}") 
        elif (lead_idx is not None) and (self.aruco_data["success"]==True):
            aruco_center=[int(self.aruco_data["position"][0]),int(self.aruco_data["position"][1])]                
            for index,id in enumerate(ids):
                x1=(tracked_bboxs[index][0])
                y1=(tracked_bboxs[index][1])
                w=(tracked_bboxs[index][2])
                h=(tracked_bboxs[index][3]) 
                if (aruco_center[0]>=x1) and (aruco_center[0]<=(x1+w)) and (aruco_center[1]>=y1) and (aruco_center[1]<=(y1+h)):
                    bbox=[[int(x1),int(y1)],[int(x1+w),int(y1+h)]]   
                    self.tracked_idx=id
                    
        #Get the bounding box coordinates of the leader
        elif lead_idx is not None and (self.aruco_data["success"]==False):                    
            x1=int(tracked_bboxs[lead_idx][0])
            y1=int(tracked_bboxs[lead_idx][1])
            w=int(tracked_bboxs[lead_idx][2])
            h=int(tracked_bboxs[lead_idx][3]) 
            bbox=[[x1,y1],[x1+w,y1+h]]
        #If leader is not visible    
        else:
            self.get_logger().info(f"Lost track of leader")
        return bbox

    def get_position(self,tracked_bbox):
        '''
        Get the distance to the person 
        Removes outliers and get the average distance to 
        the mid point of the bounding box

        Params:
        --------
        tracked box : Top left corner of bounding box,Bottom right corner of bounding box

        Returns:
        -------
        position [x,y,z] the 3D coordinates of pixels p1,p2 the 3D points in camera frame 
        
        '''
        #Clipping bbox cooridinates outside the frame
        l= (self.robot_stream_depth.shape)[0]
        w= (self.robot_stream_depth.shape)[1]
        x1=np.clip(tracked_bbox[0][0],0,w-1)
        x2=np.clip(tracked_bbox[0][1],0,w-1)
        y1=np.clip(tracked_bbox[1][0],0,l-1)
        y2=np.clip(tracked_bbox[1][1],0,l-1)
        x= int((x1+x2)/2)
        y= int((y1+y2)/2)       
        depth=self.robot_stream_depth[y,x]             
        pixel_point=[float(x),float(y)]
        position=rs.rs2_deproject_pixel_to_point(self.depth_intrinsic ,pixel_point,depth)
        prin="Pixel point x "+str(pixel_point[0])+"y "+str(pixel_point[1])+"depth "+str(depth)
        # cv2.putText(self.robot_stream_colour, prin, (10,30),cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,2550,0),2)
        
        return position

    def coordinate_transform(self,position):
        '''
        Getting the static transform between the camera and base link
        Getting the transform only once because it static and no need
        to check often

        Args:
        position: The position coodinates in camera_link frame

        Returns:
        position: The psotion coodinates in base_link frame
        '''        
        if not self.transform_acquired_base_camera:
            counter=0
            #Tries 10 times to get the transfomation 
            while (not self.transform_acquired_base_camera) and (counter<10) :
                try:
                    counter+=1
                    self.tranformer=self.tf_buffer.lookup_transform(source_frame=self.frame_source,target_frame=self.frame_target,time=rclpy.time.Time())
                    self.get_logger().info(f'Static transform from base_link to camera aquired ')                
                    self.transform_acquired_base_camera=True
                except TransformException as ex:
                    self.get_logger().info(f"Could to find tf transform {ex}")
        else:
            r_x=self.tranformer.transform.rotation.x
            r_y=self.tranformer.transform.rotation.y
            r_z=self.tranformer.transform.rotation.z
            r_w=self.tranformer.transform.rotation.w
            x=self.tranformer.transform.translation.x
            y=self.tranformer.transform.translation.x
            z=self.tranformer.transform.translation.z

            rot=np.array([r_x,r_y,r_z,r_w])
            trans=np.array([x,y,z])
            rotation_mat=np.asarray((Rotation.from_quat(rot)).as_matrix())
            homo_transform=np.hstack((np.vstack((rotation_mat,[0.0,0.0,0.0])),np.vstack((trans.reshape(3,1),1.0))))
            self.get_logger().info(f"In the transform")

            return (np.delete(np.dot(homo_transform,[position.point.x,position.point.y,position.point.z,1]),-1))      


    def imageflow_demo(self):  
        frame = self.robot_stream_colour
        if (self.robot_stream_colour) is not None:
            outputs, img_info = self.predictor.inference(frame, self.timer)
            if outputs[0] is not None:
                online_targets = self.tracker.update(outputs[0], [img_info['height'], img_info['width']], self.exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        self.results.append(
                            f"{self.frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                self.timer.toc()



                tracked_bbox=self.check_leader(online_tlwhs, online_ids)
                #Leader is available publish leader pose 
                if len(tracked_bbox)>0:
                    tracked_leader_image=self.robot_stream_colour[tracked_bbox[0][0]:tracked_bbox[0][1],tracked_bbox[1][0]:tracked_bbox[1][1]]
                    self.publisher_tracked_person_image_raw.publish(self.cvbridge.cv2_to_imgmsg(tracked_leader_image))
                    cv2.rectangle(self.robot_stream_colour,tracked_bbox[0],tracked_bbox[1], (255,0,0), 2, 1)
                    
                    tracked_position=self.get_position(tracked_bbox)
                    center=(int((tracked_bbox[0][0]+tracked_bbox[1][0])/2),int((tracked_bbox[0][1]+tracked_bbox[1][1])/2))
                    cv2.circle(self.robot_stream_colour, center, 4, (0,255,2), 2)
                    poss="3D pos o"+str(round(tracked_position[0],3))+" 1 "+str(round(tracked_position[1],3))+" 2 "+str(round(tracked_position[2],3))
                    # cv2.putText(self.robot_stream_colour, poss, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,2550,0),2)

                    #In the camera Frame Z in the X direction in robot coordinate system 
                    #Here we take that into account
                    self.tracked_person_position.header.frame_id = self.frame_target
                    self.tracked_person_position.point.x = tracked_position[2]
                    self.tracked_person_position.point.y = tracked_position[0]
                    self.tracked_person_position.point.z = tracked_position[1]                
                    ##Check if the transformation is available 

                    if self.transform_acquired_base_camera:

                        transformed_point=self.coordinate_transform(self.tracked_person_position)
                        self.get_logger().info(f"Got transform")

                        self.tracked_person_position.point.x = transformed_point[0]
                        self.tracked_person_position.point.y = transformed_point[1]
                        self.tracked_person_position.point.z = transformed_point[2]

                        print_pos="Transformed X:"+str(round(transformed_point[0],2))+"Y: "+str(round(transformed_point[1],2))+"Z: "+str(round(transformed_point[2],2))
                        # cv2.putText(self.robot_stream_colour, print_pos, (10,200),cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,2550,0),2)

                    else:
                        transformed_point=self.coordinate_transform(self.tracked_person_position)
                        self.get_logger().info(f"Transform not acquired")
      
                    self.publisher_tracked_person_position.publish(self.tracked_person_position)
                


                else:
                    cv2.putText(self.robot_stream_colour, "Leader not found", (100,80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)



            else:
                self.timer.toc()
            cv2.imshow("Tracked",self.robot_stream_colour)
            if self.record_video:
                self.video_recorder.write(self.robot_stream_colour)
            cv2.waitKey(1)
            

        else:
            pass
        self.frame_id += 1





def main(args=None):
    
    #Start of ROS node

    rclpy.logging._root_logger.log(
        'Starting person tracking  ...',
        LoggingSeverity.INFO
    )
    rclpy.init(args=args)


    library_path=get_path()
    experiment_file=osp.join(library_path,'exps/example/mot/yolox_tiny_mix_det.py')
    model_file=osp.join(library_path,'pretrained/bytetrack_tiny_mot17.pth.tar')
    device="gpu"
    fp16=True
    fuse=True
   
    args=ByteParams(exp_file=experiment_file,
                    model_file=model_file,
                    device=device,
                    fp16=fp16,
                    fuse=fuse)
    exp = get_exp(exp_file=args.exp_file,exp_name=args.name)
  
    #Start of ByteTracker
    output_dir = osp.join(exp.output_dir, exp.exp_name)
    #os.makedirs(output_dir, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    rclpy.logging._root_logger.log(
    "Model Summary: {}"+str(get_model_info(model, exp.test_size)),
    LoggingSeverity.INFO
    )
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        rclpy.logging._root_logger.log(
        "YoloX checkpoint loading started",
        LoggingSeverity.INFO
        )
       
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        rclpy.logging._root_logger.log(
        "YoloX checkpoint loading finished",
        LoggingSeverity.INFO
        )
    if args.fuse:
        rclpy.logging._root_logger.log(
        "Fusing model",
        LoggingSeverity.INFO
        )        
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        rclpy.logging._root_logger.log(
        "Using TensorRT to inference",
        LoggingSeverity.INFO
        )
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)


    tracker_byte = TrackerByte(predictor,exp,args)
    rclpy.spin(tracker_byte)





if __name__ == "__main__":  
    main()
