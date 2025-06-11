import os
import sys
import yaml
import cv2
# from cv2.aruco import estimatePoseBoard
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime

ROBOT_ONLINE = True  # Set to False if the robot is not online

CALIBRATION = True # Set to True if you want to perform calibration

def collection():
    """
    Collects data for hand-eye calibration by capturing images and detecting ArUco markers.
    """
    global ROBOT_ONLINE

    if ROBOT_ONLINE:
        # === Initialize ===
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        from controller import UR5_controller
        ur5 = UR5_controller()
        tcp_pose = ur5.rtde_r.getActualTCPPose()
        position = tcp_pose[:3]
        rotvec = tcp_pose[3:6]
        R_base2gripper = R.from_rotvec(rotvec).as_matrix()

        k4a = ur5.k4a
        capture = k4a.get_capture()
        color_image = capture.color[:, :, :3]
        # depth_image = capture.transformed_depth

    else:
        color_image = cv2.imread("hand_eye_calib_images/color_20250610_150352.png")
        cv2.imshow("Image", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # exit("Robot is not online. Using a sample image instead.")

    # === detect ArUco GridBoard ===
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()

    # Create GridBoard（3x4）
    board = cv2.aruco.GridBoard(
        size=(3, 4),              
        markerLength=0.05,       # Every marker is 5cm
        markerSeparation=0.005,  # interval is 0.5cm
        dictionary=aruco_dict
    )
    print("ArUco GridBoard created with 4x3 markers.")
    # exit("ArUco GridBoard created with 4x3 markers.")

    # detect
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        print("No ArUco marker detected. Sample not saved.")
        sys.exit(1)
    if len(ids) < 12:
        print(f"Warning: Only detected {len(ids) if ids is not None else 0} markers. May be insufficient.")
        sys.exit(1)
    # estimate pose
    data = np.load("calib_intrinsic_result.npz")
    camera_matrix = data['K']
    dist_coeffs = data['D']
    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None)
    image_markers = cv2.aruco.drawDetectedMarkers(color_image.copy(), corners, ids)
    
    if not retval:
        print("ArUco board detected but pose estimation failed.")
        sys.exit(1)
    else:
        cv2.drawFrameAxes(image_markers, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
        cv2.imshow("ArUco Detection", image_markers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("ArUco board detected and pose estimated successfully.")
        # exit()
    
    if ROBOT_ONLINE:
        # === Save Images ===
        img_dir = "hand_eye_calib_images"
        os.makedirs(img_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # cv2.imwrite(f"{img_dir}/color_{timestamp}.png", color_image)
        cv2.imwrite(f"{img_dir}/color_{timestamp}.png", image_markers)
        # cv2.imwrite(f"{img_dir}/depth_{timestamp}.png", depth_image)
        print("Image Saved.")
    
    # print(rvec)
    # print(tvec)
    R_cam2target, _ = cv2.Rodrigues(rvec)
    t_cam2target = tvec.reshape(-1)
    # print("R_cam2target:\n", R_cam2target)
    # print("t_cam2target:\n", t_cam2target)

    # exit()

    # === write YAML file ===
    yaml_path = "handeye_data.yaml"
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = {
            "R_base2gripper": [],
            "t_base2gripper": [],
            "R_cam2target": [],
            "t_cam2target": []
        }

    data["R_base2gripper"].append(R_base2gripper.tolist())
    data["t_base2gripper"].append(position)
    data["R_cam2target"].append(R_cam2target.tolist())
    data["t_cam2target"].append(t_cam2target.tolist())

    with open(yaml_path, "w") as f:
        yaml.dump(data, f)

    print(f"Sample saved to {yaml_path}. Current total: {len(data['t_base2gripper'])}")


def rotation_error_deg(R1, R2):
    R_diff = R1 @ R2.T
    angle_rad = np.arccos(np.clip((np.trace(R_diff) - 1) / 2.0, -1, 1))
    return np.degrees(angle_rad)


def calibration():
    """
    Performs hand-eye calibration using the collected data.
    """
    with open("handeye_data.yaml", "r") as f:
        data = yaml.safe_load(f)
    # target is the board
    R_base2gripper = [np.array(R) for R in data["R_base2gripper"]]

    # print(R_base2gripper[0])
    # exit()
    t_base2gripper = [np.array(t) for t in data["t_base2gripper"]]

    R_gripper2base = []
    t_gripper2base = []
    T_base2gripper_list = []
    for r,t in zip(R_base2gripper,t_base2gripper):
        T_base2gripper = np.eye(4)
        T_base2gripper[:3,:3] = r
        T_base2gripper[:3,3] = t
        T_base2gripper_list.append(T_base2gripper)
        T_gripper2base = np.linalg.inv(T_base2gripper)
        R_gripper2base.append(T_gripper2base[:3,:3])
        t_gripper2base.append(T_gripper2base[:3,3])
        

    R_cam2target = [np.array(R) for R in data["R_cam2target"]]
    print("R_cam2target:", R_cam2target[3])
    t_cam2target = [np.array(t) for t in data["t_cam2target"]]
    print("t_cam2target:", t_cam2target[3])

    # exit()

    R_target2cam = []
    t_target2cam = []
    T_target2cam_list = []
    for r,t in zip(R_cam2target, t_cam2target):
        T_cam2target = np.eye(4)
        T_cam2target[:3,:3] = r
        T_cam2target[:3,3] = t
        T_target2cam = np.linalg.inv(T_cam2target)
        T_target2cam_list.append(T_target2cam)
        R_target2cam.append(T_target2cam[:3,:3])
        t_target2cam.append(T_target2cam[:3,3])

    R_gripper2target, t_gripper2target = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    T_gripper2target = np.eye(4)
    T_gripper2target[:3, :3] = R_gripper2target
    T_gripper2target[:3, 3] = t_gripper2target.flatten()
    T_target2gripper = np.linalg.inv(T_gripper2target)
    # print(T_target2gripper)

    # R_cam2base, t_cam2base, R_target2gripper, t_target2gripper = cv2.calibrateRobotWorldHandEye(
    #     R_target2cam, t_target2cam,     # This is A: Camera observes target
    #     R_gripper2base, t_gripper2base, # This is B: Robot base to gripper
    #     method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH # SHAH is a common method for this solver
    # )

    # print("R_target2gripper:", R_target2gripper)
    # print("t_target2gripper:", t_target2gripper)

    T_base2cam_0 = T_base2gripper_list[0] @ T_gripper2target @ T_target2cam_list[0]
    print("T_base2cam:\n", T_base2cam_0)
    R_base2cam_ref = T_base2cam_0[:3, :3]
    t_base2cam_ref = T_base2cam_0[:3, 3]

    for i in range(len(T_base2gripper_list)):
        T_est = T_base2gripper_list[i] @ T_gripper2target @ T_target2cam_list[i]
        R_est, t_est = T_est[:3, :3], T_est[:3, 3]
        
        rot_err = rotation_error_deg(R_est, R_base2cam_ref)
        trans_err = np.linalg.norm(t_est - t_base2cam_ref)

        print(f"Pose {i}: rotation error = {rot_err:.3f} deg, translation error = {trans_err:.3f} m")
    T_base2cam = T_base2gripper_list[3] @ T_gripper2target @ T_target2cam_list[3]
    print("Final T_base2cam:\n", T_base2cam)

if __name__ == "__main__":

    if not CALIBRATION:
        collection()
    else:
        calibration()


# (calibrate_env) mainuser@mainuser:~/UR5_yc_mh/RTDE_UR5$ /home/mainuser/UR5_yc_mh/env/calibrate_env/bin/python3 /home/mainuser/UR5_yc_mh/RTDE_UR5/handeye_calibration.py
# R_cam2target: [[-0.18497348  0.98271865  0.00699084]
#  [-0.92063975 -0.17079113 -0.35107384]
#  [-0.34381283 -0.07137539  0.93632168]]
# t_cam2target: [-0.07178378  0.13203476  0.37125884]
# T_base2cam:
#  [[ 0.01802415 -0.8643879   0.50250243 -0.46610083]
#  [-0.99983573 -0.01462241  0.01070992 -0.28820847]
#  [-0.00190973 -0.50261292 -0.86450946  0.53767147]
#  [ 0.          0.          0.          1.        ]]
# Pose 0: rotation error = 0.000 deg, translation error = 0.000 m
# Pose 1: rotation error = 0.280 deg, translation error = 0.001 m
# Pose 2: rotation error = 0.063 deg, translation error = 0.000 m
# Pose 3: rotation error = 0.162 deg, translation error = 0.001 m
# Pose 4: rotation error = 0.253 deg, translation error = 0.003 m
# Pose 5: rotation error = 0.304 deg, translation error = 0.001 m
# Pose 6: rotation error = 0.458 deg, translation error = 0.001 m
# Pose 7: rotation error = 0.330 deg, translation error = 0.000 m
# Pose 8: rotation error = 0.368 deg, translation error = 0.002 m
# Pose 9: rotation error = 0.413 deg, translation error = 0.005 m
# Pose 10: rotation error = 0.334 deg, translation error = 0.002 m
# Pose 11: rotation error = 0.253 deg, translation error = 0.002 m
# Pose 12: rotation error = 0.476 deg, translation error = 0.003 m
# Pose 13: rotation error = 0.408 deg, translation error = 0.002 m
# Pose 14: rotation error = 0.355 deg, translation error = 0.002 m
# Pose 15: rotation error = 0.442 deg, translation error = 0.002 m
# Pose 16: rotation error = 0.240 deg, translation error = 0.001 m
# Pose 17: rotation error = 0.353 deg, translation error = 0.001 m
# Pose 18: rotation error = 0.312 deg, translation error = 0.001 m
# Pose 19: rotation error = 0.475 deg, translation error = 0.003 m
# Final T_base2cam:
#  [[ 0.02083776 -0.86426403  0.50260668 -0.46570169]
#  [-0.99978126 -0.01711065  0.01202747 -0.28900924]
#  [-0.00179498 -0.50274737 -0.86443153  0.5371946 ]
#  [ 0.          0.          0.          1.        ]]