#!/usr/bin/env python3
def getXYZ(my_pcl,x,y):
    '''
    https://pastebin.com/i71RQcU2
    https://github.com/stereolabs/zed-ros-wrapper/issues/370
    '''
    arrayPosition=y*my_pcl.row_step + x*my_pcl.point_step
    arrayPosX = arrayPosition + my_pcl.fields[0].offset
    arrayPosY = arrayPosition + my_pcl.fields[1].offset
    arrayPosZ = arrayPosition + my_pcl.fields[2].offset

    X = my_pcl.data[arrayPosX]
    Y = my_pcl.data[arrayPosY]
    X =0.0
    Y= 0.0
    Z =my_pcl.data[arrayPosZ]


    return [X,Y,Z]


# the anwe ight be at 

# https://github.com/IntelRealSense/realsense-ros/issues/1524



# import rospy
# from cv_bridge import CvBridge, CvBridgeError
# import sensor_msgs.msg  
# import numpy as np
# from PIL import Image
# i = 0
# root='/home/yu'
# def convert_depth_image(ros_image):
#     bridge = CvBridge()
#     global i
#     depth_image = bridge.imgmsg_to_cv2(ros_image, desired_encoding="passthrough")
#     depth_array = np.array(depth_image, dtype=np.float32)
#     im = Image.fromarray(depth_array)
#     im = im.convert("L")
#     idx = str(i).zfill(4)
#     im.save(root+"/depth/frame{index}.png".format(index = idx))
#     i += 1
#     print("depth_idx: ", i)

# def pixel2depth():
#     rospy.init_node('pixel2depth',anonymous=True)
#     rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", sensor_msgs.msg.Image,callback=convert_depth_image, queue_size=10)
#     rospy.spin()

# if __name__ == '__main__':
#     pixel2depth()


# bridge = CvBridge()
# cv_img = bridge.imgmsg_to_cv2(ros_img, "passthrough")
# depth_array = np.array(cv_img, dtype=np.uint16)*0.001