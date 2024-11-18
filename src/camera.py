#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError


class Camera():
    """!
    @brief      This class describes a camera.
    """

    # This defines different colors information in HSV
    COLOR_DEFINITIONS = {
        'RED': np.array([(165,50,80), (179, 255, 255)],dtype=np.uint8),
        'ORANGE': np.array([(2,70,80), (15, 255, 255)],dtype=np.uint8),
        'YELLOW': np.array([(16,100,90), (30, 255, 255)],dtype=np.uint8),
        'GREEN': np.array([(50,50,55), (90, 255, 255)],dtype=np.uint8),
        'BLUE': np.array([(95,50,50), (108, 255, 200)],dtype=np.uint8),
        'PURPLE': np.array([(109,30,20), (165, 255, 255)],dtype=np.uint8)
    }

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections = np.array([])
        """ constant locations"""
        self.april_tag_locations = np.array([
            [-250, -25],
            [250, -25],
            [250, 275],
            [-250, 275]
        ])
        self.board_corners = np.array([
            [-595.6, -185, 0],
            [595.6, -185, 0],
            [595.6, 485, 0],
            [-595.6, 485, 0]
        ])
        ''' detected apriltag locations'''
        self.image_tag_locations = np.zeros([4,2])  # The centre points of apriltags will be store by the order of ID
        """ camera matrix"""
        self.distortionMatrix = np.array([])
        self.rgbHomographyMatrix = np.array([])
        self.rgbhomographyMatrixInv = np.eye(3)
        self.depthHomographyMatrix = np.array([])
        self.depthHomographyMatrixInv = np.array([])
        """ calibrate"""
        self.loadManualCameraCalibration()


    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)
    
    def loadManualCameraCalibration(self):
        """
        This is for initialize matrixs to avoid error
        """
        # load factory intrinsic
        self.intrinsic_matrix = np.array([
            [892.877, 0, 656.044],
            [0, 893.414, 383],
            [0, 0, 1]
        ])

        # load checkboard distorion (The result is better if I don't add this)
        # self.distortionMatrix = np.array([0.129320, -0.204180, 0.003536, -0.009412, 0.000000], dtype=np.float32)

        # Manually measure extrinsic_matrix parameters
        xRot = 172                      # Rotation about x axis (degree)
        xRot = xRot * np.pi / 180.0     # Rotation about x axis (radius)
        xTrans = 20                     # Transition along x axis (mm)
        yTrans = 187                    # Transition along y axis (mm)
        zTrans = 1000                   # Transition along z axis (mm)

        # load extrinsic
        self.extrinsic_matrix = np.array([
            [1, 0, 0, xTrans],
            [0, np.cos(xRot), np.sin(xRot), yTrans],
            [0, np.sin(xRot), np.cos(xRot), zTrans],
            [0, 0, 0, 1]
        ])

    def loadCameraCalibration(self):
        """!
        @brief      Load camera intrinsic and extrinsic matrix from file. # Just manually input

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        # Automatically get extrinsic matrix
        self.solveExtrinsicPnP()

    def blockDetector(self):    # The result of this function is really bad
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        # Convert image to HSV
        rgbImage = self.GridFrame.copy()
        hsvImage = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2HSV)

        uvBlockInfo = []
        worldBlockInfo = []

        # for each color
        for color_name, (lower, upper) in self.COLOR_DEFINITIONS.items():
            # find in range
            if color_name == "RED":
                #red wraps HSV
                thresh = cv2.inRange(hsvImage,lower,upper) | cv2.inRange(hsvImage,np.array([0,40,40]),np.array([3,255,255]))
            else:
                thresh = cv2.inRange(hsvImage, lower, upper)
            
            mask = np.zeros_like(thresh)
            # mask the board
            cv2.rectangle(mask, (100,12),(1170,700), 255, cv2.FILLED)
            # mask the arm
            cv2.rectangle(mask, (550,390),(723,720), 0, cv2.FILLED)

            # morphological operation to remove noise
            kernel = np.ones((6, 6), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            #threshold is an AND between inside the board, outside the arm, and in the color range
            thresh = cv2.bitwise_and(thresh, mask)
            # cv2.imshow(f"Threshold for: {color_name}", thresh)

            # Find contours of the blocks
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contourArea = cv2.contourArea(contour)
                # Ignore contours that are too small or big
                if contourArea > 700 and contourArea < 5000:

                    #get convexity defects
                    convexHull = cv2.convexHull(contour, returnPoints = False)
                    convexityDefects = cv2.convexityDefects(contour, convexHull)

                    concave = False
                    for i in range(convexityDefects.shape[0]):
                        s,e,f,d = convexityDefects[i,0]
                        if d > 20*256:
                            # start = tuple(contour[s][0])
                            # end = tuple(contour[e][0])
                            # far = tuple(contour[f][0])
                            # cv2.line(TestImage,start,end,[0,255,0],2)
                            concave = True
                    
                    rect = cv2.minAreaRect(contour)
                    _, (width, height), _ = rect
                    aspect_ratio = min(width, height) / max(width, height)

                    #filter out concave shapes and rectangles   
                    if concave is False and aspect_ratio < 1.45 and aspect_ratio > 0.65:
                        minRect =  cv2.minAreaRect(contour)
                        if(contourArea > 650 and contourArea < 1200):
                            uvBlockInfo.append((color_name, minRect, "SMALL"))
                        elif contourArea > 1500 and contourArea < 5000:
                            uvBlockInfo.append((color_name, minRect, "LARGE"))
        print(uvBlockInfo)
        # Get world center and height of blocks
        for color_name, minRect, size in uvBlockInfo:
            # get box points
            boxCorners = np.intp(cv2.boxPoints(minRect))
            # block centers
            centerX = np.mean(boxCorners, axis=0)[0]
            centerY = np.mean(boxCorners, axis=0)[1]
            # block angle
            theta = minRect[2]
            # use inverse homography on the center point to get the unwarped image point
            unwarpedCenterPoint = self.homographyUnwarpPoints([centerX, centerY])
            # get depth
            depth = self.DepthFrameRaw[int(unwarpedCenterPoint[1])][int(unwarpedCenterPoint[0])]
            # depth = self.getWarpedDepth()[int(unWarpedCenterPoint[1])][int(unWarpedCenterPoint[0])]
            worldPoint = self.imageToWorld(unwarpedCenterPoint[0], unwarpedCenterPoint[1], depth)
            worldBlockInfo.append((color_name, minRect, size, worldPoint))

        # Draw detected blocks
        for color_name, minRect, size, worldPoint in worldBlockInfo:
            # get box points
            boxCorners = np.intp(cv2.boxPoints(minRect))
            # block centers
            centerX = np.mean(boxCorners, axis=0)[0]
            centerY = np.mean(boxCorners, axis=0)[1]
            # block angle
            theta = minRect[2]
            # draw info
            cv2.drawContours(self.GridFrame,[boxCorners], 0, (255,255,0), 2)
            cv2.circle(self.GridFrame, (int(centerX), int(centerY)), 3, (0, 0, 255), -1)
            cv2.putText(self.GridFrame, f"{color_name}", (boxCorners[0][0], boxCorners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(self.GridFrame, f"th:{int(theta)}", (boxCorners[3][0], boxCorners[3][1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        print(worldBlockInfo)
        return worldBlockInfo

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame and
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        self.GridFrame = self.VideoFrame.copy()
        self.calcRGBHomographyMatrix()

        # Find location and draw circles
        for x in self.grid_x_points:
            for y in self.grid_y_points:
                uvPoint = self.worldToImage(x, y, 0)
                cv2.circle(self.GridFrame, (uvPoint[0][0], uvPoint[1][0]), 4, (255, 255, 0), -1)
                
        # Warp image to orthogonal perspective
        if self.rgbHomographyMatrix is not None:
            self.GridFrame = self.homographyWarpFrame(self.GridFrame)
     
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here

        # Get the information of all the apriltags
        for detection in msg.detections:
            # Store and draw all the center points
            self.image_tag_locations[detection.id-1] = [detection.centre.x, detection.centre.y]
            cv2.circle(modified_image, (int(detection.centre.x), int(detection.centre.y)), 6, (255, 255, 0), -1)    # centre_coordnate only accept int

            # Draw the contour
            cv2.rectangle(modified_image, (int(detection.corners[0].x-10), int(detection.corners[0].y+10)), (int(detection.corners[2].x+10), int(detection.corners[2].y-10)), (255, 255, 0), 3)

            # Put text on it
            cv2.putText(modified_image, "ID:" + str(detection.id), [int(detection.centre.x+30), int(detection.centre.y-30)], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
        self.TagImageFrame = modified_image

    # Below are some extra functions added by author to utilize
    def imageToWorld(self, u, v, d):
        cameraPosition = (d * np.linalg.inv(self.intrinsic_matrix)) @ np.array([[u], [v], [1]])
        worldPosition = np.linalg.inv(self.extrinsic_matrix) @ np.vstack((cameraPosition, [1]))
        return worldPosition[0:3]
    
    def worldToImage(self, x, y, z):
        cameraPosition = self.extrinsic_matrix @ np.array([[x], [y], [z], [1]])
        uv = 1/cameraPosition[2] * self.intrinsic_matrix @ cameraPosition[0:3]
        return np.array(uv[0:2, :], np.int32)
    
    def solveExtrinsicPnP(self):
        # Solve the extrinsic matrix by using pnp and tag locations
        [_, Rot, T] = cv2.solvePnP(np.hstack(((self.april_tag_locations),(np.array([[0], [0], [0], [0]])))).astype("float32"), np.array(self.image_tag_locations[:,0:2], dtype=np.float32), self.intrinsic_matrix, self.distortionMatrix)

        # Transfer rotate vector to rotate matrix
        R, _ = cv2.Rodrigues(Rot)
        
        # Output the extrinsic_matrix
        self.extrinsic_matrix = np.vstack(((np.hstack((R, T))), ([0, 0, 0, 1])))
    
    def calcRGBHomographyMatrix(self):
        # Put points in the same plane
        sourcePoints = np.array([])
        for tag in self.image_tag_locations:
            if sourcePoints.any():
                sourcePoints = np.vstack((sourcePoints, self.imageToWorld(tag[0], tag[1], 1000).T[:,0:2]))
            else:
                sourcePoints = self.imageToWorld(tag[0], tag[1], 1000).T[:,0:2]
        sourcePoints = np.array(sourcePoints, dtype=np.float32)
        destPoints = np.array(self.april_tag_locations, dtype=np.float32)

        # Find homogeneous matrix
        if sourcePoints.any():
            self.rgbHomographyMatrix = cv2.findHomography(sourcePoints, destPoints)[0]

    def homographyWarpFrame(self, frame):
        # Perform homography warp on rgb frame
        if self.rgbHomographyMatrix.any():
            return cv2.warpPerspective(frame, self.rgbHomographyMatrix, (1280,720))
    
    def homographyUnwarpPoints(self, point):
        # Perform point unwarp on rgb frame
        if self.rgbHomographyMatrix is not None:
            self.rgbhomographyMatrixInv = np.linalg.inv(self.rgbHomographyMatrix)
        return cv2.perspectiveTransform(np.array([[point]]), self.rgbhomographyMatrixInv)[0][0][:]

class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image


class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                self.camera.blockDetector() # add blockdetector in gridframe
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()