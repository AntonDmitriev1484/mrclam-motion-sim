import numpy as np
import matplotlib.pyplot as plt
from load_data import * 

T=100

def measured_vo_to_pose(gt, vo):
    x = gt[0].x
    y = gt[0].y
    o = gt[0].orientation
    positional_data=[]
    dT = 1/T
    for row in vo:
        fv = row.fv
        dy = fv * dT * math.sin(o)         # sin = O/H
        dx = fv * dT * math.cos(o)        # cos = A/H
        x += dx
        y += dy

        av = row.av #delta orientation
        o += av * dT

        positional_data.append(Pose(time=row.time, x=x, y=y, orientation=o))
    return positional_data

def measured_vo_to_fakeSLAM(gt_pose, mes_pose, mes_vo, SLAM_T, T):
    # May need to add some artificial gaussian error into this later,
    # to replicate imperfect pose estimation from the map?
    dT = 1/T
    fakeSLAM_pose = [gt_pose[0]]
    for gt, mes, vo in zip(gt_pose, mes_pose, mes_vo):
        if mes.time % SLAM_T == 0:
            fakeSLAM_pose.append(gt)
        else:
            prev_pose = fakeSLAM_pose[-1]
            dy = vo.fv * dT * math.sin(prev_pose.orientation)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.orientation)        # cos = A/H
            do = vo.av * dT
            cur_pose = Pose(mes.time, prev_pose.x + dx, prev_pose.y + dy, prev_pose.orientation + do)
            fakeSLAM_pose.append(cur_pose)

    return fakeSLAM_pose

# algo1: Simple trig approximation

def rotate_vector(vector, degrees):
    r = np.radians(degrees)
    rotation_matrix = np.array([[np.cos(r), -np.sin(r)],
                                [np.sin(r),  np.cos(r)]])
    return np.dot(vector, rotation_matrix)

DBG = False
def dv(start, end, plt, label=None, color='000000'):
        if DBG == True: return plt.arrow(start[0], start[1], (end-start)[0], (end-start)[1], color=color, label=label, head_width=0.001, head_length=0.001, width=0.0001)
        else: return None

def trig_approx(ref_pose, true_pose, imu_pose, imu_dir, seg_delta_angle, plt=None):
        
        # dv(imu_pose, imu_pose + 0.5*(imu_dir)/np.linalg.norm(imu_dir), plt, color='red' )

        # A - B creates an arrow from tip of B to tip of A
        UWB_range = np.linalg.norm(ref_pose - true_pose) # defines a circle radius # Might be signage on this
        # dv(true_pose, ref_pose, plt, color='blue', label='UWB_range')

        v_0 = imu_pose - ref_pose # Vector from ref to imu
        # dv(ref_pose, ref_pose+v_0, plt, color='blue')

        v_UWB = UWB_range * (v_0 / np.linalg.norm(v_0)) # Vector from imu to edge of the UWB circle, in the direction of IMU

        v_1 = v_0 - v_UWB

        pivot = ref_pose + v_UWB # v_1 and v_2 both rotate about this point

        # dv(pivot, pivot + v_1, plt, color='purple')

        # dv(imu_pose, imu_pose+(0.5*rotate_vector(v_0, 90)/np.linalg.norm(v_0)), plt, color='blue')

        A = np.arccos(np.dot(imu_dir,v_1)/ (np.linalg.norm(imu_dir)*np.linalg.norm(v_1))) # IN RADIANS
        S = np.dot(rotate_vector(v_1, 90), imu_dir) / np.linalg.norm(np.dot(rotate_vector(v_1, 90), imu_dir))
        B = (A-np.radians(90))/abs(A-np.radians(90)) # 90 is 1.5708 radians
        # print(f"A {np.degrees(A)} S {S} B {B}")

        rot_angle = -B*S*A
        if B<0: rot_angle = -B*S*(np.radians(90)-A)
        # print(f"rot_angle {np.degrees(rot_angle)}")

        v_2 = rotate_vector(v_1, np.degrees(-B*S*A))
        if B<0: v_2 = rotate_vector(v_1, np.degrees(-B*S*(1.5708-A)))

        TURN_CEIL = 0.10745999999999996
        print(f"seg_delta_angle {seg_delta_angle}")
        v_2 *= (seg_delta_angle / TURN_CEIL)**5

        # print(v_2) # SO even when we hard set this to 0, it still skews? Why???
        # Now need to somehow scale v_2 by seg_delta_angle
        # if seg_delta_angle is 0, we want v_2 to be 0
        # otherwise, we want v_2 to max out at 1, lets assume 1.708 rad (90 degree) is the max turning we can do in one segment
        # I think this threshold should scale with our range_T, longer range_T means higher turn ceiling.

        # dv(pivot, pivot+v_2, plt, color='orange')

        v_steer = (pivot + v_2) - ref_pose
        # dv(ref_pose , ref_pose + v_steer, plt, color='orange')

        estimate_point = UWB_range * (v_steer / np.linalg.norm(v_steer))
        
        # if DBG: plt.scatter(estimate_point[0], estimate_point[1], color = 'pink', s=20)
        return estimate_point

def measured_vo_to_algo1(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
    # cluster SLAM_T means the period that SLAM is run on any client.
    # compute path for robot_id.
    robot_id-=1 # Note robot_id should be between 0 and 4

    DBG = False
    dT = 1/T

    # sim_time = min( [len(data) for data in all_gt_pose] )
    sim_time = min( [len(all_gt_pose[0]), len(all_gt_pose[1]) ] )

    approx_pose = [all_gt_pose[robot_id][0]] #starts at ground truth

    # dbg_view = sim_time
    dbg_view = 120*100 
    # dbg_view = 61 # range_T = 30
    ref_id = 1 

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5,3.5)
    ax.set_ylim(-3.5,0.5)
    ax.grid(True)

    theta_adjust = 0
    sum_delta_angle = 0

    for t in range(1,dbg_view):
        # If its time for some client in our cluster to get slammed

        if t % SLAM_T == 0:
            approx_pose.append(all_gt_pose[robot_id][t])
        elif t % range_T == 0:
            imu_pose = np.array([ approx_pose[-1].x, approx_pose[-1].y ])
            imu_dir_abs_radians = approx_pose[-1].orientation

            imu_dir = np.array([1*math.cos(imu_dir_abs_radians), 1*math.sin(imu_dir_abs_radians)]) # Need this vector to be imu_pose+imu_dir?
            true_pose = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            ref_pose = np.array((0,0))

            predict_point = trig_approx(ref_pose, true_pose, imu_pose, imu_dir, sum_delta_angle, plt)
            # final_predict = predict_point

            seg_start_point = np.array((approx_pose[-range_T].x, approx_pose[-range_T].y)) # Get the pose 200ms in the past
            seg_end_point = np.array((approx_pose[-1].x ,approx_pose[-1].y)) # Latest pose from normal IMU integration is our end point

            final_predict = predict_point


            do = (vo.av) * dT
            pose_estimate = Pose(t, final_predict[0], final_predict[1], imu_dir_abs_radians + do)
            approx_pose.append(pose_estimate)
            sum_delta_angle = 0
        else:
            # Otherwise just append VO data
            vo = all_mes_vo[robot_id][t]
            prev_pose = approx_pose[-1]

            # Somehow we're getting way denser points ~2x than VO integration, why? we aren't moving forward enough?
            dy = vo.fv * dT * math.sin(prev_pose.orientation)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.orientation)        # cos = A/H
            do = (vo.av) * dT
            cur_pose = Pose(t, prev_pose.x + dx, prev_pose.y + dy, prev_pose.orientation + do)
            
            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is

            approx_pose.append(cur_pose)

    x, y = ([p.x for p in approx_pose[:dbg_view]] , [p.y for p in approx_pose[:dbg_view]])
    plt.scatter(x, y, c='blue', s=1)

    x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_view]] , [p.y for p in all_gt_pose[robot_id][:dbg_view]])
    plt.scatter(x, y, c='green', s=1)

    # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # plt.scatter(x, y, c='green', s=1)
    x, y = ([p.x for p in mes_pose[robot_id][:dbg_view]] , [p.y for p in mes_pose[robot_id][:dbg_view]])
    plt.scatter(x, y, c='red', s=1)

    plt.show()

    return approx_pose

def pf_pose_estimate(uwb_range, ref_pose, imu_path, imu_pose):
    return None

