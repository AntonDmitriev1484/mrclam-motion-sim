import matplotlib
# matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


from utils import *
from load_data import * 
from pf import ParticleFilter

T=100
fig, ax = plt.subplots()
ax.set_xlim(-0.5,3.5)
ax.set_ylim(-3.5,0.5)
ax.grid(True)
UNCERTAIN = 0.01 # 10cm

def run_pf(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
    # This algorithm will use a particle filter to estimate point location during each range.

    robot_id=0
    dT = 1/T
    sim_time = min( [len(all_gt_pose[0]), len(all_gt_pose[1]) ] )

    start_pose = all_gt_pose[robot_id][0] #starts at ground truth
    imu_segment = [ State(0,0,0) for i in range(0, range_T)] # An array of States
    imu_segment[0] = State(start_pose.x, start_pose.y, start_pose.orientation)
    last_imu_integration = State(start_pose.x, start_pose.y, start_pose.orientation)

    # Segment from 1 to 1.5 minutes has problems
    # dbg_start = 40 * 100
    dbg_start = 0
    dbg_start = 0 * 100
    # dbg_end = 200*100
    # dbg_end = 120 * 100
    dbg_end = 300 * 100
    # dbg_end = 80 * 100
    dbg_show_particles = True

    dbg_view_T = 10*100

    sum_delta_angle = 0

    estimated_poses = np.zeros((1,3))
    start_pose = all_gt_pose[robot_id][0]
    estimated_poses[0] = np.array([start_pose.x, start_pose.y, start_pose.orientation])
    
    full_poses = np.zeros((1,4))
    full_poses[0] = np.array([0, start_pose.x, start_pose.y, start_pose.orientation])
    prev_pf_pose = full_poses[0]

    anchors = np.zeros((4, 2))
    anchors[0, :] = np.array([0,0])

    pf = ParticleFilter(1000)
    pf.generate(estimated_poses[0])
    ref_pos = np.array((0,0))

    resample_count = 0
    range_count = 0

    timer = 0

    for t in range(0,dbg_end):

        if t % range_T == 0:
            print(f" Range # {t/range_T}")

            true_pos = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            v_uwb = true_pos - ref_pos

            AoA_precision = np.pi
            GT_orientation = all_gt_pose[robot_id][t].orientation
            
            pf.measurement(true_pos, anchors, AoA_precision, GT_orientation)

            estimate = pf.estimate()
            prev_pf_pose = estimate

            # Debug draw orientation vector to see where cloud is trending
            start = estimate[[X,Y]]
            diff = np.array([0.1*math.cos(estimate[[O]]), 0.1*math.sin(estimate[[O]])])
            # dv(estimate[[X,Y]], estimate[[X,Y]]+diff)

            estimated_poses = np.append(estimated_poses, [estimate], axis = 0)
            full_poses = np.append(full_poses, np.array([np.array([t, estimate[0], estimate[1], estimate[2]])]), axis = 0)

            if pf.need_resample(sum_delta_angle): 
                pf.resample()
                resample_count+=1

            sum_delta_angle = 0
            range_count+=1

        else:
            # Otherwise perform regular imu integration
            i = t%range_T
            vo = all_mes_vo[robot_id][t]
            pf.update(vo) 

            prev_pose = imu_segment[i-1]

            dy = vo.fv * dT * math.sin(prev_pose.o)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.o)        # cos = A/H
            do = (vo.av) * dT
            cur_pose = State(prev_pose.x + dx, prev_pose.y + dy, prev_pose.o + do)


            zzz = prev_pf_pose + np.array([dx, dy, do])
            asdf = np.array([np.array([t, zzz[X], zzz[Y], zzz[O]])])
            full_poses = np.append(full_poses, asdf, axis = 0)
            prev_pf_pose = zzz

            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is
            imu_segment[i] = cur_pose


        if dbg_show_particles and dbg_start < t and t % dbg_view_T ==0:
            dparticle_weights(pf.particles)
    

    print(f" Out of {range_count} ranges, {resample_count} were resamples")

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Robot trajectory for t={dbg_end/(T)} seconds')

    # Estimated poses has less than all_gt_pose because its jsut the pf estimates so dbg_staert and dbg_end are out of bounds
    x, y = (estimated_poses[:,X], estimated_poses[:,Y])
    plt.scatter(x, y, c='blue', s=10)
    
    x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_end]] , [p.y for p in all_gt_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='green', s=1)

    # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # plt.scatter(x, y, c='green', s=1)
    x, y = ([p.x for p in mes_pose[robot_id][:dbg_end]] , [p.y for p in mes_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='red', s=1)

    plt.show()

    # return estimated_poses
    return full_poses
