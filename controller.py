import numpy as np
from rtde_receive import RTDEReceiveInterface
from rtde_control import RTDEControlInterface
import pyrealsense2 as rs 
import robotiq_gripper
import pyk4a
from pyk4a import Config, PyK4A

class UR5_controller():
    def __init__(self):
        self.ROBOT_HOST = "192.168.20.25"
        self.scale_factor = 0.5 # Scale factor for velocity command
        self.rtde_r, self.rtde_c, self.gripper = self.initialize_robot()
        self.k4a = self.initialize_camera()
        self.realsense = self.initialize_hand_camera()

        self.gripper_position = self.gripper.get_current_position()
        self.gripper_max = self.gripper.get_max_position()
        self.gripper_min = self.gripper.get_min_position()

        # Can use to check if the grasping is success or not.
        self.is_gripping = self.gripper.is_gripping()

    def initialize_robot(self): 
        rtde_r = RTDEReceiveInterface(self.ROBOT_HOST)
        rtde_c = RTDEControlInterface(self.ROBOT_HOST)
        print("Creating gripper...")
        gripper = robotiq_gripper.RobotiqGripper()
        print("Connecting to gripper...")
        gripper.connect(self.ROBOT_HOST, 63352)
        print("Activating gripper...")
        gripper.activate()
        return rtde_r, rtde_c, gripper

    # Function to get robot joint state
    def get_robot_joint_state(self):
        state = np.array(self.rtde_r.getActualQ())
        action = np.array(self.rtde_r.getTargetQ())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state, gripper_state))
        action = np.concatenate((action, gripper_state))
        return state, action
    
    # Function to get robot eef state
    def get_robot_eef_state(self):
        state = np.array(self.rtde_r.getActualTCPPose())
        action = np.array(self.rtde_r. getTargetTCPPose())
        gripper_state = np.array([self.gripper.get_current_position()]) 
        state = np.concatenate((state, gripper_state))
        action = np.concatenate((action, gripper_state))
        return state, action

    # Function to get end-effector force
    def get_robot_eef_force(self):
        force = np.array(self.rtde_r.getActualTCPForce())
        return force
    
    def control_gripper(self,gripper_position):
        if gripper_position < self.gripper_min:
            gripper_position = self.gripper_min

        if gripper_position > self.gripper_max:
            gripper_position = self.gripper_max
        
        self.gripper.move(gripper_position, 155, 255)
    
    def move_by_joints(self,joint_positions:np.array[6]):
        self.rtde_c.moveJ(joint_positions)

    def move_by_tool(self, target_pose:np.array[6]):
        self.rtde_c.moveL(target_pose)
        
    
    def initialize_camera(self): 
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_720P,
                camera_fps=pyk4a.FPS.FPS_30,
                depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                synchronized_images_only= True,
            )
        )
        k4a.start()
        # Set white balance
        k4a.whitebalance = 4500
        assert k4a.whitebalance == 4500
        return k4a

    def initialize_hand_camera(self):
        realsense = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        realsense.start(config)
        return realsense