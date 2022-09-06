#!/usr/bin/env python3
def getXYZ(my_pcl,x,y):
    '''
    https://pastebin.com/i71RQcU2
    https://github.com/stereolabs/zed-ros-wrapper/issues/370
    '''
    arrayPosition=x*my_pcl.row_step + y*my_pcl.point_step
    arrayPosX = arrayPosition + my_pcl.fields[0].offset
    arrayPosY = arrayPosition + my_pcl.fields[1].offset
    arrayPosZ = arrayPosition + my_pcl.fields[2].offset
    print(f"The length of data {len(my_pcl.data)}")
    X = my_pcl.data[arrayPosX]
    Y = my_pcl.data[arrayPosY]
    Z =my_pcl.data[arrayPosZ]


    return [X,Y,Z]
