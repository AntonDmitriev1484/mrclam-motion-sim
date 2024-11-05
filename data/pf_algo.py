import numpy as np
import matplotlib.pyplot as plt
from load_data import * 

T=100
fig, ax = plt.subplots()
ax.set_xlim(-0.5,3.5)
ax.set_ylim(-3.5,0.5)
ax.grid(True)

def rotate_segment(imu_segment):
    return None

def rotate_vector(vector, degrees):
    r = np.radians(degrees)
    rotation_matrix = np.array([[np.cos(r), -np.sin(r)],
                                [np.sin(r),  np.cos(r)]])
    return np.dot(vector, rotation_matrix)

norm = np.linalg.norm
dot = np.dot

def unit(v):
    return v/norm(v)

def rad_between_vec(v_1, v_2):
    return np.arccos(dot(v_1, v_2) / (norm(v_1)*norm(v_2)))

DBG = False
def dv(start, end, label=None, color='000000'):
    if DBG == True: return plt.arrow(start[0], start[1], (end-start)[0], (end-start)[1], color=color, label=label, head_width=0.001, head_length=0.001, width=0.0001)
    else: return None

def dp(pos, label=None, color='000000'):
    if DBG == True: return plt.arrow(pos[0], pos[1], (end-start)[0], (end-start)[1], color=color, label=label, head_width=0.001, head_length=0.001, width=0.0001)
    else: return None

def State2Vec(state): # Returns <x,y>, orientation
    return np.array((state.x, state.y)), state.o

class ParticleFilter1:
    def ParticleFilter1(self, n_particles, imu_segment, uwb_ref, uwb_range):
        self.n_particles = n_particles
        self.seg = imu_segment
        self.pose = imu_segment[-1]
        self.ref = uwb_ref
        self.range = uwb_range

    def sample(self):
        return None
    
    def motion(self):
        return None
    
    def measurement(self):
        return None
    
    def resample(self):
        return None
    
    def converged(self):
        # Check convergence condition here
        return True
    
    def get_estimated_segment(self):
        return self.seg, self.pose

def measured_vo_to_algo2(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
    # This algorithm will use a particle filter to estimate point location during each range.

    robot_id=0
    dT = 1/T
    sim_time = min( [len(all_gt_pose[0]), len(all_gt_pose[1]) ] )

    ref_pos = np.array((0,0))
    poses = [all_gt_pose[robot_id][0]] #starts at ground truth

    # dbg_view = sim_time
    dbg_view = 120*100 
    # dbg_view = 61 # range_T = 30

    sum_delta_angle = 0

    estimated_poses = []
    imu_segment = [ 0 for i in range(0, range_T)] # An array of States

    for t in range(1,dbg_view):
        # If its time for some client in our cluster to get slammed

        if t % SLAM_T == 0:
            #TODO: Do I need to add to imu_segment here also?
            estimated_poses.append(all_gt_pose[robot_id][t])
        elif t % range_T == 0:
            # Use Particle filter here!

            true_pos = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            v_uwb = true_pos - ref_pos

            pf = ParticleFilter1(1000, imu_segment, ref_pos, norm(v_uwb))

            # imu_pose = np.array([ approx_pose[-1].x, approx_pose[-1].y ])
            # imu_dir_abs_radians = approx_pose[-1].orientation

            # imu_dir = np.array([1*math.cos(imu_dir_abs_radians), 1*math.sin(imu_dir_abs_radians)]) # Need this vector to be imu_pose+imu_dir?
            # true_pose = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])

            # predict_point = trig_approx(ref_pose, true_pose, imu_pose, imu_dir, sum_delta_angle, plt)
            # # final_predict = predict_point

            # seg_start_point = np.array((approx_pose[-range_T].x, approx_pose[-range_T].y)) # Get the pose 200ms in the past
            # seg_end_point = np.array((approx_pose[-1].x ,approx_pose[-1].y)) # Latest pose from normal IMU integration is our end point

            # final_predict = predict_point

            # do = (vo.av) * dT
            # pose_estimate = Pose(t, final_predict[0], final_predict[1], imu_dir_abs_radians + do)
            # approx_pose.append(pose_estimate)

            estimated_poses.append(imu_segment)
            sum_delta_angle = 0
            
        else:
            # Otherwise perform regular imu integration
            vo = all_mes_vo[robot_id][t]
            prev_pose = imu_segment[t-1]

            dy = vo.fv * dT * math.sin(prev_pose.o)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.o)        # cos = A/H
            do = (vo.av) * dT
            cur_pose = State(prev_pose.x + dx, prev_pose.y + dy, prev_pose.orientation + do)
            
            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is
            imu_segment[t] = cur_pose

    x, y = ([p.x for p in estimated_poses[:dbg_view]] , [p.y for p in estimated_poses[:dbg_view]])
    plt.scatter(x, y, c='blue', s=1)

    x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_view]] , [p.y for p in all_gt_pose[robot_id][:dbg_view]])
    plt.scatter(x, y, c='green', s=1)

    # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # plt.scatter(x, y, c='green', s=1)
    x, y = ([p.x for p in mes_pose[robot_id][:dbg_view]] , [p.y for p in mes_pose[robot_id][:dbg_view]])
    plt.scatter(x, y, c='red', s=1)

    plt.show()

    return approx_pose

