import os
import os.path as osp
import sys
import math
import numpy as np
import cv2
import torch
from PIL import ImageFont
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # convert sensor messages
# import pyrealsense2 as rs

##Need this to fix the relative import issue
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))



from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression

# Global parameteres
ROOT = os.path.dirname(os.path.realpath(__file__))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class Inferer:
    def __init__(self, weights, device, yaml_file, img_size, half, conf_thres, iou_thres, classes, agnostic_nms, max_det, hide_labels, hide_conf): 
        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        print(f"The CUDA is true {cuda}")
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml_file)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size

        self.conf_thres=conf_thres
        self.iou_thres=iou_thres
        self.classes=classes
        self.agnostic_nms=agnostic_nms
        self.max_det=max_det
        self.hide_labels=hide_labels
        self.hide_conf=hide_conf

        # Half precision
        if half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        # Init the webcamera
        self.camera = cv2.VideoCapture(0)
        _, self.stream = self.camera.read()
        self._start_time = time.time()


    
        

      


        

    def infer_video(self):
        ''' Model Inference and results visualization '''

        while abs(self._start_time-time.time())<1000:
            _, stream = self.camera.read()
            img, img_src = self.precess_image(stream, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            pred_results = self.model(img)
            det = non_max_suppression(pred_results, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, self.max_det)[0]
            img_ori = img_src

            # check image and font
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            font_path=osp.join(ROOT, 'yolov6/utils/Arial.ttf')
            self.font_check(font=font_path)

            if len(det):
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()

                for *xyxy, conf, cls in reversed(det):      
                    class_num = int(cls)  # integer class
                    label = None if self.hide_labels else (self.class_names[class_num] if self.hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                    self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=self.generate_colors(class_num, True))

                img_src = np.asarray(img_ori)
                cv2.imshow('frame', img_src)
   
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()

    def infer_ros_video(self):
        ''' Infer the date '''

        # while abs(self._start_time-time.time())<1000:
        stream = self.robot_stream
        img, img_src = self.precess_image(stream, self.img_size, self.stride, self.half)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim
        pred_results = self.model(img)
        det = non_max_suppression(pred_results, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, self.max_det)[0]
        img_ori = img_src

        # check image and font
        assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
        font_path=osp.join(ROOT, 'yolov6/utils/Arial.ttf')
        self.font_check(font=font_path)

        if len(det):
            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()

            for *xyxy, conf, cls in reversed(det):      
                class_num = int(cls)  # integer class
                label = None if self.hide_labels else (self.class_names[class_num] if self.hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=self.generate_colors(class_num, True))

            img_src = np.asarray(img_ori)
            cv2.imshow('frame', img_src)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        cv2.destroyAllWindows()

    # def infer_realsense_video(self,depth=False):
    #     ''' Model Inference and results visualization with intel camera '''
    #     pipe = rs.pipeline()
    #     configure = rs.config()
    #     pipe.start(configure)
    #     align_to = rs.stream.color
    #     align = rs.align(align_to)
        
    #     while abs(self._start_time-time.time())<1000:
    #         temp = pipe.wait_for_frames()
    #         aligned_frames = align.process(temp)
    #         stream = np.asanyarray(aligned_frames.get_color_frame().get_data(),dtype=np.uint8) 
    #         depth_frame = np.asanyarray(aligned_frames.get_depth_frame().get_data(),dtype=np.uint8)
    #         input_image=stream
    #         img, img_src = self.precess_image(input_image, self.img_size, self.stride, self.half)
    #         img = img.to(self.device)
    #         if len(img.shape) == 3:
    #             img = img[None]
    #             # expand for batch dim
    #         pred_results = self.model(img)
    #         det = non_max_suppression(pred_results, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, self.max_det)[0]
    #         img_ori = img_src

    #         # check image and font
    #         assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
    #         font_path=osp.join(ROOT, 'YOLOv6/yolov6/utils/Arial.ttf')
    #         self.font_check(font=font_path)

    #         if len(det):
    #             det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()

    #             for *xyxy, conf, cls in reversed(det):                   
    #                 class_num = int(cls)  # integer class
    #                 label = None if self.hide_labels else (self.class_names[class_num] if self.hide_conf else f'{self.class_names[class_num]} {conf:.2f}')

    #                 self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=self.generate_colors(class_num, True))

    #             img_src = np.asarray(img_ori)
    #             cv2.imshow('frame', img_src)
    #             if depth==True:
    #                 cv2.imshow('depth',depth_frame)
   
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break

    #     cv2.destroyAllWindows()
 
    @staticmethod
    def precess_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        try:
            # img_src = cv2.imread(path)
            assert img_src is not None, f'Invalid image'
        except Exception as e:
            LOGGER.warning(e)
        image = letterbox(img_src, img_size, stride=stride)[0]

        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)

    @staticmethod
    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color


# def test_intelrealsense(depth=True):
#     '''
#     Testing the intel real sense camera
#     '''

#     pipe = rs.pipeline()
#     configure = rs.config()
#     pipe.start(configure)
#     align_to = rs.stream.color
#     align = rs.align(align_to)

#     while True:
#         temp = pipe.wait_for_frames()
#         aligned_frames = align.process(temp)
#         depth_frame = np.asanyarray(aligned_frames.get_depth_frame().get_data(),dtype=np.uint8)
#         frame = np.asanyarray(aligned_frames.get_color_frame().get_data(),dtype=np.uint8)        
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
        
#         cv2.imshow('test_web',frame)
#         cv2.imshow('test_depth',depth_frame)
#     cv2.destroyAllWindows()

        



def test_webcam():
    '''
    Check
    '''
    cam=cv2.VideoCapture(3)   
    while True:
        ret, frame = cam.read()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.imshow('test_web',frame)
    cv2.destroyAllWindows()

    
def run_stream(vid_source=0,depth=False):

    #Initiate the inferer and the paramerters

    weights=osp.join(ROOT, 'weights/yolov6n.pt')
    device = 'gpu'
    yaml = osp.join(ROOT,'data/coco.yaml')
    img_size=640
    half=False
    classes=0 ## Classify only people 
    conf_thres=0.25
    iou_thres=0.45
    agnostic_nms=False
    max_det=1000
    hide_labels=False
    hide_conf=False



    ##Intialise the inferer
    inferer = Inferer(weights, device, yaml, img_size, half, conf_thres, iou_thres, classes, agnostic_nms, max_det, hide_labels, hide_conf)
    if vid_source==1:
        inferer.infer_realsense_video(True)
    else:
        inferer.infer_ros_video()



def main(args=None):
    rclpy.init(args=args)

    weights=osp.join(ROOT, 'weights/yolov6n.pt')
    device = 'gpu'
    yaml = osp.join(ROOT,'data/coco.yaml')
    img_size=640
    half=False
    classes=0 ## Classify only people 
    conf_thres=0.25
    iou_thres=0.45
    agnostic_nms=False
    max_det=1000
    hide_labels=False
    hide_conf=False

    minimal_subscriber = Inferer(weights, device, yaml, img_size, half, conf_thres, iou_thres, classes, agnostic_nms, max_det, hide_labels, hide_conf)
    minimal_subscriber.infer_video()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()
    


if __name__=='__main__':
    main()