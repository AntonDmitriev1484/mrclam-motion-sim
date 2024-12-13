import matplotlib
# matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random


from utils import *
from load_data import * 
from pf import ParticleFilter1, ParticleFilter2
from antcolony_pf import AntColonyParticleFilter
from dual_pf import DualMeasurementParticleFilter

T=100
fig, ax = plt.subplots()
ax.set_xlim(-0.5,3.5)
ax.set_ylim(-3.5,0.5)
ax.grid(True)
UNCERTAIN = 0.01 # 10cm


def run_original_pf(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
    # This algorithm will use a particle filter to estimate point location during each range.

    robot_id=0
    dT = 1/T
    sim_time = min( [len(all_gt_pose[0]), len(all_gt_pose[1]) ] )

    ref_pos = np.array((0,0))
    start_pose = all_gt_pose[robot_id][0] #starts at ground truth
    imu_segment = [ State(0,0,0) for i in range(0, range_T)] # An array of States
    imu_segment[0] = State(start_pose.x, start_pose.y, start_pose.orientation)
    last_imu_integration = State(start_pose.x, start_pose.y, start_pose.orientation)

    # Segment from 1 to 1.5 minutes has problems
    # dbg_start = 40 * 100
    dbg_start = 0
    # dbg_end = 300 * 100
    dbg_end = 120 * 100
    dbg_view_T = 10 * 100


    sum_delta_angle = 0

    estimated_poses = []
    S_vectors = []
    trig_estimated_poses = [] # trig approx poses based on particle filter estimates

    pf = ParticleFilter1(2000) # Can't really observe behavior on 2k particles freezes plot
    pf.generate(all_gt_pose[robot_id][0])

    for t in range(0,dbg_end):
        # if t % SLAM_T == 0:
        #     #TODO: Do I need to add to imu_segment here also?
        #     estimated_poses.append(all_gt_pose[robot_id][t])

        if t % range_T == 0:
            print(f" Range # {t/range_T}")

            true_pos = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            v_uwb = true_pos - ref_pos

            pf.measurement(ref_pos, norm(v_uwb), true_pos, sum_delta_angle)


            estimate = pf.estimate()

            if len(estimated_poses)==0: est_dir2 = np.array([1*math.cos(estimate[O]), 1*math.sin(estimate[O])])
            else: est_dir2 = (estimate[:O] - estimated_poses[-1][:O])

            estimated_poses.append(estimate)
            
            # est_dir = np.array([1*math.cos(estimate[O]), 1*math.sin(estimate[O])])
            # imu_dir_abs_radians = last_imu_integration.o
            # imu_dir = np.array([1*math.cos(imu_dir_abs_radians), 1*math.sin(imu_dir_abs_radians)]) 

            vec = trig_approx(ref_pos, true_pos, estimate[[X,Y]], est_dir2, sum_delta_angle)
            S_vectors.append(vec) # For later re-drawing

            hint = unit(true_pos - estimate[[X,Y]]) # Suppose we know the general direction that we drifted off of GT

            # Performs about as bad as usual when we pass our trigapprox vector in as a hint
            # -> the one thats basically wrong half the time
            if pf.need_resample(sum_delta_angle): pf.resample(sum_delta_angle, hint)
            # Now rotate the segment to meet this estimate

            sum_delta_angle = 0
        else:
            # Otherwise perform regular imu integration
            i = t%range_T
            vo = all_mes_vo[robot_id][t]
            pf.update(vo) # Could we always just take the distribution mean from here to determine our motion?

            prev_pose = imu_segment[i-1]

            dy = vo.fv * dT * math.sin(prev_pose.o)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.o)        # cos = A/H
            do = (vo.av) * dT
            cur_pose = State(prev_pose.x + dx, prev_pose.y + dy, prev_pose.o + do)
            last_imu_integration=cur_pose

            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is
            imu_segment[i] = cur_pose

        # if t % dbg_view_T ==0:
        #     dparticle_weights(pf.particles)
    # plt.show()

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    # Estimated poses has less than all_gt_pose because its jsut the pf estimates so dbg_staert and dbg_end are out of bounds
    x, y = ([p[0] for p in estimated_poses[:dbg_end]] , [p[1] for p in estimated_poses[:dbg_end]])
    plt.scatter(x, y, c='blue', s=10)
    
    x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_end]] , [p.y for p in all_gt_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='green', s=1)

    # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # plt.scatter(x, y, c='green', s=1)
    x, y = ([p.x for p in mes_pose[robot_id][:dbg_end]] , [p.y for p in mes_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='red', s=1)

    plt.show()
    # plt.savefig('paths.png')

    return estimated_poses

def run_pf2(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
    # This algorithm will use a particle filter to estimate point location during each range.

    robot_id=0
    dT = 1/T
    sim_time = min( [len(all_gt_pose[0]), len(all_gt_pose[1]) ] )

    ref_pos = np.array((0,0))
    start_pose = all_gt_pose[robot_id][0] #starts at ground truth
    imu_segment = [ State(0,0,0) for i in range(0, range_T)] # An array of States
    imu_segment[0] = State(start_pose.x, start_pose.y, start_pose.orientation)
    last_imu_integration = State(start_pose.x, start_pose.y, start_pose.orientation)

    # Segment from 1 to 1.5 minutes has problems
    # dbg_start = 40 * 100
    dbg_start = 0
    dbg_end = 300 * 100
    # dbg_end = 120 * 100
    dbg_view_T = 10*100

    hint_imu = np.array([all_gt_pose[robot_id][0].x, all_gt_pose[robot_id][0].y, 0])
    hint_T = 3*range_T

    sum_delta_angle = 0

    estimated_poses = []
    S_vectors = []
    trig_estimated_poses = [] # trig approx poses based on particle filter estimates

    pf = ParticleFilter2(2000) # Can't really observe behavior on 2k particles freezes plot
    pf.generate(all_gt_pose[robot_id][0])
    ref_pos2 = np.array((0,-5))
    ref_pos = np.array((0,0))

    resample_count = 0
    range_count = 0

    for t in range(0,dbg_end):
        # if t % SLAM_T == 0:
        #     #TODO: Do I need to add to imu_segment here also?
        #     estimated_poses.append(all_gt_pose[robot_id][t])

        if t % range_T == 0:
            print(f" Range # {t/range_T}")

            true_pos = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            v_uwb = true_pos - ref_pos

            pf.measurement(ref_pos, norm(v_uwb), sum_delta_angle)

            estimate = pf.estimate()
            estimated_poses.append(estimate)
            
            if pf.need_resample(sum_delta_angle): 
                pf.resample()
                resample_count+=1

            sum_delta_angle = 0
            range_count+=1

            
            if t % hint_T ==0:
                # I suspect the estimate orientation we're outputting is wrong
                # Because orientation has no tie to the actual state estimation
                # So thats why the hint_imu point is starting so far out.
                # we could do estimate[t] - estimate[t-1] to figure out the orientation to start integrating at
                # l = 0.05
                # vec_ori_est = [estimate[X] + l*math.cos(estimate[O]) , estimate[Y] + l*math.sin(estimate[O])]
                # dv(estimate[[X,Y]], vec_ori_est)
                # dv(hint_imu[[X,Y]], estimate[[X,Y]]) # This is a pass by reference apparently
                # est_vec = estimated_poses[len(estimated_poses)-1] - estimated_poses[len(estimated_poses)-2]
                hint_imu = np.copy(estimate)

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

            delta = [vo.fv * dT * math.cos(hint_imu[[O]]), vo.fv * dT * math.sin(hint_imu[[O]]) , (vo.av) * dT]
            hint_imu[[X,Y,O]] += delta

            # dp(hint_imu[[X,Y]], color='orange')
            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is
            imu_segment[i] = cur_pose


        # if t % dbg_view_T ==0:
        #     dparticle_weights(pf.particles)
    # plt.show()

    print(f" Out of {range_count} ranges, {resample_count} were resamples")

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    # Estimated poses has less than all_gt_pose because its jsut the pf estimates so dbg_staert and dbg_end are out of bounds
    x, y = ([p[0] for p in estimated_poses[:dbg_end]] , [p[1] for p in estimated_poses[:dbg_end]])
    plt.scatter(x, y, c='blue', s=10)
    
    x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_end]] , [p.y for p in all_gt_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='green', s=1)

    # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # plt.scatter(x, y, c='green', s=1)
    x, y = ([p.x for p in mes_pose[robot_id][:dbg_end]] , [p.y for p in mes_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='red', s=1)

    plt.show()
    # plt.savefig('paths.png')

    return estimated_poses

def run_antcolony_pf(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
    # This algorithm will use a particle filter to estimate point location during each range.

    robot_id=0
    dT = 1/T
    sim_time = min( [len(all_gt_pose[0]), len(all_gt_pose[1]) ] )

    ref_pos = np.array((0,0))
    start_pose = all_gt_pose[robot_id][0] #starts at ground truth
    imu_segment = [ State(0,0,0) for i in range(0, range_T)] # An array of States
    imu_segment[0] = State(start_pose.x, start_pose.y, start_pose.orientation)
    last_imu_integration = State(start_pose.x, start_pose.y, start_pose.orientation)

    # Segment from 1 to 1.5 minutes has problems
    # dbg_start = 40 * 100
    dbg_start = 0
    # dbg_end = 300 * 100
    dbg_end = 120 * 100
    dbg_view_T = 10 * 100

    sum_delta_angle = 0

    estimated_poses = []
    S_vectors = []
    trig_estimated_poses = [] # trig approx poses based on particle filter estimates

    pf = AntColonyParticleFilter(100, 20)

    # 
    pf.generate(all_gt_pose[robot_id][0])

    for t in range(0,dbg_end):
        # if t % SLAM_T == 0:
        #     #TODO: Do I need to add to imu_segment here also?
        #     estimated_poses.append(all_gt_pose[robot_id][t])

        if t % range_T == 0:
            print(f" Range # {t/range_T}")

            true_pos = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            v_uwb = true_pos - ref_pos

            pf.measurement(ref_pos, norm(v_uwb))

            estimate = pf.estimate()
            estimated_poses.append(estimate)

            if pf.need_resample(): pf.resample()

            sum_delta_angle = 0
        else:
            # Otherwise perform regular imu integration
            i = t%range_T
            vo = all_mes_vo[robot_id][t]
            pf.update(vo) # Could we always just take the distribution mean from here to determine our motion?

            prev_pose = imu_segment[i-1]

            dy = vo.fv * dT * math.sin(prev_pose.o)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.o)        # cos = A/H
            do = (vo.av) * dT
            cur_pose = State(prev_pose.x + dx, prev_pose.y + dy, prev_pose.o + do)
            last_imu_integration=cur_pose

            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is
            imu_segment[i] = cur_pose

        if t % dbg_view_T ==0:
            dparticle_weights(pf.particles)

    # plt.show()

    # Estimated poses has less than all_gt_pose because its jsut the pf estimates so dbg_staert and dbg_end are out of bounds
    x, y = ([p[0] for p in estimated_poses[:dbg_end]] , [p[1] for p in estimated_poses[:dbg_end]])
    plt.scatter(x, y, c='blue', s=10)

    x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_end]] , [p.y for p in all_gt_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='green', s=1)

    # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # plt.scatter(x, y, c='green', s=1)
    x, y = ([p.x for p in mes_pose[robot_id][:dbg_end]] , [p.y for p in mes_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='red', s=1)

    plt.show()
    # plt.savefig('paths.png')

    return estimated_poses

def run_dual_pf(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
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
    dbg_end = 300 * 100
    # dbg_end = 120 * 100
    dbg_view_T = 10 * 100


    sum_delta_angle = 0

    estimated_poses = []
    S_vectors = []
    trig_estimated_poses = [] # trig approx poses based on particle filter estimates

    ref_pos = np.array((0,0))
    ref2_pos = np.array((0,-5))
    pf = DualMeasurementParticleFilter(2000) # Can't really observe behavior on 2k particles freezes plot
    pf.generate(all_gt_pose[robot_id][0])

    for t in range(0,dbg_end):
        if t % SLAM_T == 0:
            true_pos = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            
            pf.dual_measurement(ref_pos, norm(true_pos-ref_pos), ref2_pos, norm(true_pos-ref2_pos))
            # dparticle_weights(pf.particles)

            estimated_poses.append(all_gt_pose[robot_id][t])
        elif t % range_T == 0:
            print(f" Range # {t/range_T}")

            true_pos = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            v_uwb = true_pos - ref_pos

            pf.measurement(ref_pos, norm(v_uwb))


            estimate = pf.estimate()

            estimated_poses.append(estimate)
            
            if pf.need_resample(sum_delta_angle): pf.resample(sum_delta_angle)

            sum_delta_angle = 0
        else:
            # Otherwise perform regular imu integration
            i = t%range_T
            vo = all_mes_vo[robot_id][t]
            pf.update(vo) # Could we always just take the distribution mean from here to determine our motion?

            prev_pose = imu_segment[i-1]

            dy = vo.fv * dT * math.sin(prev_pose.o)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.o)        # cos = A/H
            do = (vo.av) * dT
            cur_pose = State(prev_pose.x + dx, prev_pose.y + dy, prev_pose.o + do)
            last_imu_integration=cur_pose

            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is
            imu_segment[i] = cur_pose

        # if t % dbg_view_T ==0:
        #     dparticle_weights(pf.particles)
    # plt.show()

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')

    # Estimated poses has less than all_gt_pose because its jsut the pf estimates so dbg_staert and dbg_end are out of bounds
    x, y = ([p[0] for p in estimated_poses[:dbg_end]] , [p[1] for p in estimated_poses[:dbg_end]])
    plt.scatter(x, y, c='blue', s=10)
    
    x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_end]] , [p.y for p in all_gt_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='green', s=1)

    # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # plt.scatter(x, y, c='green', s=1)
    x, y = ([p.x for p in mes_pose[robot_id][:dbg_end]] , [p.y for p in mes_pose[robot_id][:dbg_end]])
    plt.scatter(x, y, c='red', s=1)

    plt.show()
    # plt.savefig('paths.png')

    return estimated_poses
