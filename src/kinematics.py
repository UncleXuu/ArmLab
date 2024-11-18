"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    T = np.eye(4) # Create initial matrix
    theta1, theta2, theta3, theta4, theta5 = joint_angles # get the current theta angles

    dh_params[0,0] = theta1
    dh_params[3,0] = -theta2
    dh_params[5,0] = -theta3
    dh_params[6,0] = -theta4
    dh_params[8,0] = theta5
    
    if link == 1:
        dh_params = dh_params[0,:]
        theta, d, a, alpha = dh_params
        T = get_transform_from_dh(a, alpha, d, theta)
        
        return T
    
    elif link == 2:
        dh_params = dh_params[0:4,:]
    elif link == 3:
        dh_params = dh_params[0:6,:]
    elif link == 4:
        dh_params = dh_params[0:7,:]
    elif link >= 5:
        dh_params = dh_params
    
    for theta, d, a, alpha in dh_params:
        T0 = get_transform_from_dh(a, alpha, d, theta)
        T = T @ T0

    return T


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix T from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transformation matrix.
    """

    T = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)], 
                  [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)], 
                  [0, np.sin(alpha), np.cos(alpha), d], 
                  [0, 0, 0, 1]])

    return T


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix. // Revolute to Z axis first, then Y and X.

                TODO: Implement this function return the 3 Euler angles from a 4x4 transformation matrix T
                If you like, add an argument to specify the Euler angles used (xyx, zyz, etc.)

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    phi = np.arctan2(T[1, 0], T[0, 0])  # Rotation around Z (YAW)
    theta = np.arctan2(-T[2, 0], np.sqrt(T[2, 1]**2 + T[2, 2]**2))  # Rotation around Y (PITCH)
    psi = np.arctan2(T[2, 1], T[2, 2])  # Rotation around X (ROLL)

    return [phi, theta, psi]


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the 6DOF pose vector from a 4x4 transformation matrix T

    @param      T     transformation matrix

    @return     The pose vector from T.
    """
    position = T[:3,-1]
    euler_angles = get_euler_angles_from_T(T)
    pose = np.hstack((position, euler_angles))

    return pose


def FK_pox(joint_angles, m_mat, s_lst): # No need to edit
    """!
    @brief      Get a  representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4x4 homogeneous matrix representing the pose of the desired link

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4x4 homogeneous matrix representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v): # No need to edit
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(pose, vertical = True, horizontal = False):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose vector as np.array to joint angles

    @param      dh_params  The dh parameters // Do not need to input dh parameters, instead embed these
                                                parameters into this function
    @param      pose       The desired pose vector as np.array 

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """
    # Extract position and orientation from the pose vector
    px, py, pz, yaw, pitch, roll = pose
    Pe = np.array([px, py, pz])

    # Compute theta1
    theta1 = np.arctan2(-Pe[0], Pe[1])
    theta1 = clamp(theta1)

    # Compute the wrist center position
    a5 = 174.15    # Length from joint 3 to end-effector
    # Pw = Pe - a5 * np.array([-np.sin(theta1)*np.cos(roll), np.cos(theta1)*np.cos(roll), -np.sin(roll)])
    if pz < 0:
        pz = 0.1                # Add constraint in z axis

    # If the arm is in high position, make it pick horizontally
    if horizontal:
        if theta1 >= 0:
            z = np.array([np.sin(theta1), -np.cos(theta1), 0])
        elif theta1 < 0:
            z = np.array([np.sin(theta1), -np.cos(np.abs(theta1)), 0])

    # If the arm is in low position, make it pick vertical
    elif vertical:
        z = np.array([0, 0, 1]) # To make the arm pick vertically

    else:
        if theta1 >= 0:
            z = np.array([np.sin(theta1)*np.sin(np.pi/4), -np.cos(theta1)*np.sin(np.pi/4), np.sin(np.pi/4)])
        elif theta1 < 0:
            z = np.array([np.sin(theta1)*np.sin(np.pi/4), -np.cos(np.abs(theta1))*np.sin(np.pi/4), np.sin(np.pi/4)])
    
    Pw = Pe + a5 * z


    # Offsets from DH parameters
    d1 = 103.91    # Base height from Link 1
    a2 = 200       # Length from joint 2 to joint 3
    a3 = 50        # Length from joint 3 to joint 4
    a4 = 200       # Length from joint 4 to joint 5

    # Compute r and s
    r = np.sqrt(Pw[0]**2 + Pw[1]**2)
    s = Pw[2] - d1

    # Compute the distance between joint 2 and the wrist center
    D = np.sqrt(r**2 + s**2)

    # Law of Cosines for theta3
    theta = 1.818   # Radians of original angle
    cos_theta_theta3 = ((a2**2 + a3**2) + a4**2 - D**2) / (2 * np.sqrt(a2**2 + a3**2) * a4)
    if abs(cos_theta_theta3) > 1.0:
        raise ValueError("Position is unreachable.")
    theta_theta3 = np.arccos(cos_theta_theta3)
    theta3 = clamp(theta-theta_theta3)
    # Recompute sin and cos for the chosen theta3
    sin_theta_theta3 = np.sin(theta_theta3)

    # Compute theta2
    alpha = np.arctan2(a4 * sin_theta_theta3, 206.155 - a4 * cos_theta_theta3)
    beta = np.arctan2(r, s)
    gamma = 0.245   # The angle to adjust
    theta2 = beta - alpha - gamma

    # Normalize angles
    theta2 = clamp(theta2)

    # Compute theta4 and theta5
    # theta4  = roll - theta2 - theta3

    # Make sure the wrist is vertical or horizontal
    if horizontal:
        theta4 = theta - gamma - theta2 - theta3 - np.pi/2
        theta5 = np.pi + theta1 - yaw
    elif vertical:
        theta4 = theta - gamma - theta2 - theta3
        theta5 = np.pi + theta1 - yaw
    else:
        theta4 = theta - gamma - theta2 - theta3 - np.pi/4
        theta5 = np.pi + theta1 - yaw

    # Normalize angles
    theta4 = clamp(theta4)
    theta5 = clamp(theta5)

    # Append the solution
    return [theta1, theta2, theta3, theta4, theta5]

