import pandas as pd
from io import FileIO
from collections import namedtuple
import math as math
import csv

DATA_DIR = "/home/admitriev/ISF24/mrclam-motion-sim/data"

from scipy.spatial.transform import Rotation as R

#Immutable by design
Pose = namedtuple('Pose', ["time","x","y","orientation"])
State = namedtuple('State', ['x','y','o']) # Pose but less verbose

Odometry = namedtuple('Odometry', ["time","fv","av"])
TUMPose = namedtuple('TUMPose', ["timestamp","tx","ty","tz","qx","qy","qz","qw"])
#https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats

def dataframe_to_posetuple(dataframe):
    return list(map(lambda row: Pose(row[0], row[1], row[2], row[3]), dataframe))

def dataframe_to_odometrytuple(dataframe):
    return list(map(lambda row: Odometry(row[0], row[1], row[2]), dataframe))

def pose_to_TUM(pose):
    # In robot case we're only rotation about Z-axis, so we change yaw
    quat = R.from_euler('xyz', [0, 0, pose.orientation]).as_quat()
    # NOTE: Hardcoded to Centiseconds, need to convert back to seconds for TUM format
    return TUMPose(float(pose.time)/100, pose.x, pose.y, 0, quat[0], quat[1], quat[2], quat[3])

def pose_to_TUM_epic(pose):
    # In robot case we're only rotation about Z-axis, so we change yaw
    quat = R.from_euler('xyz', [0, 0, pose.orientation]).as_quat()
    # NOTE: Hardcoded to Centiseconds, need to convert back to seconds for TUM format
    return TUMPose(float(pose.time), pose.x, pose.y, 0, quat[0], quat[1], quat[2], quat[3])

def write_pose_data_TUM(filename, pose_data):
    print(f"Writing data for {filename}")
    with open(f"{DATA_DIR}/eval_dataset/{filename}.txt", 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        file.write("# timestamp tx ty tz qx qy qz qw\n") #THIS FUCKING COMMENT WASTED 1 HOUR OF MY LIFE
        for row in pose_data:
            tum = pose_to_TUM(row)
            # formatted = " ".join([f"{field:.7f}" for field in tum])
            formatted = [f"{field:.4f}" for field in tum]
            # line=f"{tum.timestamp} {tum.tx} {tum.ty} {tum.tz} {tum.qx} {tum.qy} {tum.qz} {tum.qw}"
            # .rstrip
            writer.writerow(formatted)

def load_MRCLAM_groundtruth(dataset_id, robot_id):
    fstream = FileIO(f"{DATA_DIR}/MRCLAM_Dataset{dataset_id}/Robot{robot_id}_Groundtruth.dat")
    dataframe = pd.read_csv(fstream, delim_whitespace=True, comment='#').to_numpy()
    return dataframe

def load_MRCLAM_odometry(dataset_id, robot_id):
    fstream = FileIO(f"{DATA_DIR}/MRCLAM_Dataset{dataset_id}/Robot{robot_id}_Odometry.dat")
    dataframe = pd.read_csv(fstream, delim_whitespace=True, comment='#').to_numpy()
    return dataframe

def load_MRCLAM(dataset_id, robot_id):
    return (load_MRCLAM_groundtruth(dataset_id, robot_id), load_MRCLAM_odometry(dataset_id, robot_id))

# Shifts timestamps to start at 0
def normalize_time(dataframe):
    # 0 is the index of timestamp in all datasets
    start = dataframe[0][0]
    for row in dataframe:
        row[0] -= start
    return dataframe


def read_pose_data(filename):
    fstream = FileIO(f"C:\\Users\\soula\\OneDrive\\Desktop\\Programming\\ISF24\\UWBSLAMsim\\data\\processed_dataset\\{filename}.txt")
    dataframe = pd.read_csv(fstream, delim_whitespace=True, comment='#').to_numpy()
    return dataframe_to_posetuple(dataframe)


def write_pose_data(filename, pose_data):
    with open(f"C:\\Users\\soula\\OneDrive\\Desktop\\Programming\\ISF24\\UWBSLAMsim\\data\\processed_dataset\\{filename}.txt", 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(Pose._fields)
        for row in pose_data:
            writer.writerow(row)

# I don't think you ever have two datapoints occur within 10ms of each other
# so interpolate on by adding in centiseconds where two datapoints are missing centiseconds
# Want one datapoint per centisecond
def interpolate_data(dataframe, T):

    fl_data = []
    interpolated_data = []

    for row in dataframe:
        fl_data.append( row._replace(time=(math.floor(row.time*T))) )

    for row, next_row in zip(fl_data, fl_data[1:]):
        td = next_row.time - row.time
        for i in range(td):
            interpolated_data.append(row._replace(time=(row.time+i)))

    return interpolated_data
