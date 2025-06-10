import os
import sys
import yaml
import cv2
# from cv2.aruco import estimatePoseBoard
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime

ROBOT_ONLINE = True  # Set to False if the robot is not online

CALIBRATION = True  # Set to True if you want to perform calibration

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
        data = np.load("calib_intrinsic_result.npz")
        camera_matrix = data['K']
        dist_coeffs = data['D']

        # === Save Images ===
        img_dir = "hand_eye_calib_images"
        os.makedirs(img_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"{img_dir}/color_{timestamp}.png", color_image)
        # cv2.imwrite(f"{img_dir}/depth_{timestamp}.png", depth_image)
        print("Image Saved.")

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
    # estimate pose
    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, camera_matrix, dist_coeffs, None, None)
    if not retval:
        print("ArUco board detected but pose estimation failed.")
        sys.exit(1)
    R_cam2target, _ = cv2.Rodrigues(rvec)
    t_cam2target = tvec.reshape(-1)

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


def calibration():
    """
    Performs hand-eye calibration using the collected data.
    """
    with open("handeye_data_0.yaml", "r") as f:
        data = yaml.safe_load(f)
    # target is the board
    R_base2gripper = [np.array(R) for R in data["R_base2gripper"]]

    # print(R_base2gripper[0])
    # exit()
    t_base2gripper = [np.array(t) for t in data["t_base2gripper"]]

    R_gripper2base = []
    t_gripper2base = []
    for r,t in zip(R_base2gripper,t_base2gripper):
        T_base2gripper = np.eye(4)
        T_base2gripper[:3,:3] = r
        T_base2gripper[:3,3] = t
        T_gripper2base = np.linalg.inv(T_base2gripper)
        R_gripper2base.append(T_gripper2base[:3,:3])
        t_gripper2base.append(T_gripper2base[:3,3])
        

    R_cam2target = [np.array(R) for R in data["R_cam2target"]]
    t_cam2target = [np.array(t) for t in data["t_cam2target"]]

    # print(R_gripper2base[-1], t_gripper2base[-1])
    # print(T_base2gripper)
    # exit()


    # R_cam2target = [R.T for R in R_target2cam]
    # t_cam2target = [-R.T @ t for R, t in zip(R_target2cam, t_target2cam)]

    # R_base2cam, t_base2cam, _, _ = cv2.calibrateRobotWorldHandEye(
    #     R_base2gripper, t_base2gripper,
    #     R_cam2target, t_cam2target,
    #     method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
    # )

    R_base2cam, t_base2cam = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_cam2target, t_cam2target,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    print("Calibration Result:")
    print("Rotation matrix (base2cam):\n", R_base2cam)
    print("Translation vector (base2cam):\n", t_base2cam)

    # print("Rotation matrix (base2cam):\n", R_gripper2target)
    # print("Translation vector (base2cam):\n", t_gripper2target)

if __name__ == "__main__":

    if not CALIBRATION:
        collection()
    else:
        calibration()