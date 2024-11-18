# ArmLab
Armlab is a course project for UMich ROB550-Robotic Systems Lab. In this project, an RGB-D camera was used to detect the size, color, position, and height of blocks. The project also involved implementing both forward and inverse kinematics for a 5DOF robotic arm, which was then used to perform a series of tasks. This project achieved first place in the final competition.

![Stacking](/media/stacking.gif)
*Stacking in rainbow order*

![Shooting](/media/shooting.gif)
<!-- <img src="/media/shooting.gif" alt="Shooting" width="100" height="100"> -->
*Shooting a basket*

## Computer Vision
The positions of four April tags on the table were first detected, followed by using the `solvePnP` method to calculate and store the camera's extrinsic matrix. 
![apriltag_detection](/media/apriltag_detection.png)

With the April tag positions and the extrinsic matrix, the camera perspective could be wrapped so that every point on the table could be accurately mapped within the image.
![project_grid_points](/media/project_grid_points.png)

Finally, `cv2` was used to detect each block's color, position, size, and orientation.
![block_detection](/media/block_detection.png)

## Robot Kinematic
To transform the rotational information of each joint into the coordinates of the end effector, the Denavit-Hartenberg (DH) convention was used for calculation. `matplt` was used to verify this Forward Kinematic.

|![DH_convention](/media/DH_convention.png) | ![FK_verification](/media/FK_verification.png)|
|:---------------:|:-----------------------------------------:|

In Inverse Kinematic, there are more than one solution for the same end effector's location because this is just a 5DOF arm. In our situation, to achive largest workspack, for high and far blocks, the horizontal pick pose was used. For noraml situation, vertical pick was used.
![pick_poses](/media/pick_poses.png)

For more information about robot kinematic, please refer to the report.