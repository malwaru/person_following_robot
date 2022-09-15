#! /usr/bin/python
import cv2


class ArucoDetector():
    def __init__(self) -> None:
        self._robot_stream = None
        self.camera = cv2.VideoCapture(0)
        self.arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_50)
        self.arucoParams = cv2.aruco.DetectorParameters_create()
        

    def detect_marker(self):
        while True:
            _,stream = self.camera.read()
            (corners, ids, rejected) = cv2.aruco.detectMarkers(stream, self.arucoDict,parameters=self.arucoParams)
            topLeft=[20,50]
            topRight=[10,10]

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
                    # print(f"[INFO] ArUco marker ID: {markerID}")
                    # show the output image
            else:
                 cv2.putText(stream,"No marker",
                        (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            cv2.imshow("Image", stream)
            cv2.waitKey(1)
  

if __name__=='__main__':
    print("Detecting Aruco markers")    
    aruco_detector=ArucoDetector()
    aruco_detector.detect_marker()
   

 

 