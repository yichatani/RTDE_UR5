import os
import sys
import yaml
import cv2
# from cv2.aruco import estimatePoseBoard
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime

ROBOT_ONLINE = False  # Set to False if the robot is not online

CALIBRATION = False # Set to True if you want to perform calibration

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
        color_image = cv2.imread("hand_eye_calib_images/color_20250610_151329.png")
        cv2.imshow("Image", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # exit("Robot is not online. Using a sample image instead.")

    # === detect ArUco GridBoard ===
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_params = cv2.aruco.DetectorParameters()

    # Create GridBoard（4x3）
    board = cv2.aruco.GridBoard(
        size=(4, 3),              
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
        cv2.imshow("ArUco Detection", image_markers)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("ArUco board detected and pose estimated successfully.")
    
    if ROBOT_ONLINE:
        # === Save Images ===
        img_dir = "hand_eye_calib_images"
        os.makedirs(img_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{img_dir}/color_{timestamp}.png", color_image)
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
    with open("handeye_data_1.yaml", "r") as f:
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

    exit()

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