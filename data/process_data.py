from load_data import *
import numpy as np
import matplotlib.pyplot as plt

T=100

def crop_data(gt_pose, vo):
    # start and end times
    start_gt_pose = gt_pose[0][0]
    end_gt_pose = gt_pose[-1][0]
    start_vo = vo[0][0]
    end_vo = vo[-1][0]
    # If start_vo starts after start_gt_pose cut pose data to start at the same time as start_vo
    if start_gt_pose < start_vo:
        gt_pose = list(filter(lambda x: x[0]>start_vo , gt_pose))
    else:
        vo = list(filter(lambda x: x[0]>start_gt_pose , vo))
    # If vo ends after gt_pose  cut vo data to end at gt_pose data
    if end_gt_pose < end_vo:
        vo = list(filter(lambda x: x[0]<end_gt_pose , vo))
    else:
        gt_pose = list(filter(lambda x: x[0]<end_vo , gt_pose))
    return gt_pose, vo

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

DBG = True
def dv(start, end, plt, label=None, color='000000'):
        if DBG == True: return plt.arrow(start[0], start[1], (end-start)[0], (end-start)[1], color=color, label=label)
        else: return None


def trig_approx(ref_pose, true_pose, imu_pose, imu_dir, plt=None):
        
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

        # dv(pivot, pivot+v_2, plt, color='orange')

        v_steer = (pivot + v_2) - ref_pose
        # dv(ref_pose , ref_pose + v_steer, plt, color='orange')

        estimate_point = UWB_range * (v_steer / np.linalg.norm(v_steer))
        
        # if DBG: plt.scatter(estimate_point[0], estimate_point[1], color = 'pink', s=20)
        return estimate_point


def measured_vo_to_algo1(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, T, mes_pose=None):
    # cluster SLAM_T means the period that SLAM is run on any client.
    # compute path for robot_id.
    robot_id-=1 # Note robot_id should be between 0 and 4

    DBG = False
    dT = 1/T

    # sim_time = min( [len(data) for data in all_gt_pose] )
    sim_time = min( [len(all_gt_pose[0]), len(all_gt_pose[1]) ] )

    approx_pose = [all_gt_pose[robot_id][0]] #starts at ground truth

    # dbg_view = sim_time
    # dbg_view = 120*100 
    dbg_view = 300 # range_T = 30
    ref_id = 1 

    fig, ax = plt.subplots()
    ax.set_xlim(-0.5,3.5)
    ax.set_ylim(-3.5,0.5)
    ax.grid(True)

    theta_adjust = 0

    for t in range(1,dbg_view):
        # If its time for some client in our cluster to get slammed

        # if t % SLAM_T == 0:
        #     approx_pose.append(all_gt_pose[robot_id][t])
        if t % range_T == 0:
            imu_pose = np.array([ approx_pose[-1].x, approx_pose[-1].y ])
            imu_dir_abs_radians = approx_pose[-1].orientation

            imu_dir = np.array([1*math.cos(imu_dir_abs_radians), 1*math.sin(imu_dir_abs_radians)]) # Need this vector to be imu_pose+imu_dir?
            true_pose = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            ref_pose = np.array((0,0))
            # ref_pose = np.array([ all_gt_pose[ref_id][t].x , all_gt_pose[ref_id][t].y ])

            # plt.scatter(true_pose[0], true_pose[1], color='green', s=20)
            # plt.scatter(imu_pose[0], imu_pose[1], color='red', s=20)

            predict_point = trig_approx(ref_pose, true_pose, imu_pose, imu_dir, plt)

            seg_start_point = np.array((approx_pose[-range_T].x, approx_pose[-range_T].y)) # Get the pose 200ms in the past
            seg_end_point = np.array((approx_pose[-1].x ,approx_pose[-1].y)) # Latest pose from normal IMU integration is our end point

            v_a = seg_end_point - seg_start_point

            v_steer = predict_point - seg_start_point
            # SO the problem is that v_steer is super super far off because of forward drift, and it adds a ton of angular error
            
            # Scale how far our point is projected based off of how little change in angle we have
            # Multiply v_2 by angle change
            # 0 angle change, 0 outwards projection, will fix this forward drift fuckery for the initial forward motion.

            dv(seg_start_point, v_steer + seg_start_point, plt, color='purple')

            v_b = (v_steer / np.linalg.norm(v_steer)) * np.linalg.norm(v_a)

            if range_T == t:
                print(seg_start_point)
                print(seg_end_point)
                print(v_a)
                print(v_b)
                print(approx_pose)

            theta_adjust += np.arccos( np.dot(v_a, v_b) / (np.linalg.norm(v_a) * np.linalg.norm(v_b)))
            # I think the theta_adjust we're adding on to each point is probably too large.

            final_predict = seg_start_point + v_b # So that we are aligned tip-to-tail (more or less).
            dv(seg_start_point, final_predict, plt, color='blue')
            #TODO: Why is v_a the right vector to put here and not v_b?

            do = (vo.av) * dT
            pose_estimate = Pose(t, final_predict[0], final_predict[1], imu_dir_abs_radians + theta_adjust + do)
            approx_pose.append(pose_estimate)

            # prev_pose = approx_pose[-1]

            # dy = vo.fv * dT * math.sin(prev_pose.orientation)         # sin = O/H
            # dx = vo.fv * dT * math.cos(prev_pose.orientation)        # cos = A/H
            # do = vo.av * dT
            # cur_pose = Pose(t, prev_pose.x + dx, prev_pose.y + dy, prev_pose.orientation + do)
            
            # approx_pose.append(cur_pose)
        else:
            # Otherwise just append VO data
            vo = all_mes_vo[robot_id][t]
            prev_pose = approx_pose[-1]

            dy = vo.fv * dT * math.sin(prev_pose.orientation)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.orientation)        # cos = A/H
            do = (vo.av) * dT
            cur_pose = Pose(t, prev_pose.x + dx, prev_pose.y + dy, prev_pose.orientation + do)
            
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

# Testing trig_approx

# Case 1: IMU pose within UWB range
# ref_pose = np.array((0,0))
# true_pose = np.array((5,-12))
# imu_pose = np.array((1,-9))
# imu_dir =  imu_pose - np.array((0,-15))

# Case 2: IMU pose outside UWB range
# ref_pose = np.array((0,0))
# true_pose = np.array((0,-10))
# imu_pose = np.array((3,-12))
# imu_dir =  imu_pose - np.array((0,-15))

# Case 3: IMU pose outside UWB range, opposite imu motion vector
# ref_pose = np.array((0,0))
# true_pose = np.array((1,-10))
# imu_pose = np.array((3,-12))
# imu_dir =  -(imu_pose - np.array((0,-15)))

# Case 4 - 3rd to last point - see test4 screenshot
# ref_pose = np.array((0,0))
# true_pose = np.array((7,-7))
# imu_pose = np.array((5,-11))
# imu_dir =  imu_pose - np.array((11,-12))
# Yeah this works

# fig, ax = plt.subplots()
# ax.grid(True)

# plt.scatter(ref_pose[0], ref_pose[1], c='blue', s=10)
# plt.scatter(true_pose[0], true_pose[1], c='green', s=10)
# plt.scatter(imu_pose[0], imu_pose[1], c='red', s=10)

# trig_approx(ref_pose, true_pose, imu_pose, imu_dir, plt=plt)

# plt.show()



all_gt_pose = [ [] for i in range(1,6) ]
all_mes_vo = [ [] for i in range(1,6) ]
all_mes_pose = [ [] for i in range(1,6) ]


for i in range(1,3):
    gt_pose_df, vo_df = load_MRCLAM(1, i)
    gt_pose_df, vo_df = crop_data(gt_pose_df, vo_df)

    gt_pose_df = normalize_time(gt_pose_df)
    gt_pose = interpolate_data(dataframe_to_posetuple(gt_pose_df), T) # Gives us pose on order of cs
    all_gt_pose[i-1] = gt_pose

    vo_df = normalize_time(vo_df)
    mes_vo = interpolate_data(dataframe_to_odometrytuple(vo_df), T) # Gives us pose on order of cs
    mes_pose = measured_vo_to_pose(gt_pose, mes_vo)
    all_mes_pose[i-1] = mes_pose
    all_mes_vo[i-1] = mes_vo

    # SLAM_T = 30 # default edgeSLAM runtime is between 200 to 300ms
    fSLAM_pose = measured_vo_to_fakeSLAM(gt_pose, mes_pose, mes_vo, 1000, T) # Run "SLAM" every 20 cs

    # write_pose_data(f"R{i}_mes", mes_pose)
    # write_pose_data(f"R{i}_gt", gt_pose)
    # write_pose_data(f"R{i}_fSLAM", fSLAM_pose)

    # write_pose_data_TUM(f"R{i}_mes", mes_pose)
    # write_pose_data_TUM(f"R{i}_gt", gt_pose[:1000])
    write_pose_data_TUM(f"R{i}_gt", gt_pose)
    # write_pose_data_TUM(f"R{i}_fSLAM", fSLAM_pose)




# Originally had this set to 300 -> one slam every 3000ms
range_T = 30 # Ranging once every 300ms - i.e one member of the cluster gets Slammed every 300ms
# This frequency makes a big impact on how long we can track the pose with ground truth
SLAM_T = 300 # 5 robots, round robin offload on each, so robot 1 gets a SLAM result every 1500ms

for i in range(1,2):
    approx_pose =  measured_vo_to_algo1(1, all_gt_pose, all_mes_vo, range_T, T, SLAM_T, mes_pose=all_mes_pose)
    # write_pose_data_TUM(f"R{i}_alg1", approx_pose[:1000])
    write_pose_data_TUM(f"R{i}_alg1", approx_pose)