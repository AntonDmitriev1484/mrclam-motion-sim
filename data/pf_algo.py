import numpy as np
import matplotlib.pyplot as plt
import random
from load_data import * 

T=100
fig, ax = plt.subplots()
ax.set_xlim(-0.5,3.5)
ax.set_ylim(-3.5,0.5)
ax.grid(True)
UNCERTAIN = 0.01 # 10cm

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

DBG = True
def dv(start, end, label=None, color='000000'):
    end = end-start
    if DBG: plt.arrow(start[0], start[1], end[0], end[1], color=color, label=label, head_width=0.001, head_length=0.001, width=0.0001)

def dp(pos, label=None, color='000000'):
    if DBG: return plt.scatter(pos[0], pos[1], color=color, label=label,  s=0.1)
    else: return None

def dparticle(particle, color='000000'):
    if DBG:
        dp((particle.x, particle.y), color)
        d = 0.05 # length we want to show the direction vector at
        start, _ = State2Vec(particle)
        end = np.array( (particle.x + d*np.cos(particle.o), particle.y + d*np.sin(particle.o) )  )
        dv( start, end, color)
    else: return None

def State2Vec(state): # Returns <x,y>, orientation
    return np.array((state.x, state.y)), state.o

def Vec2State(state):
    return State()

class ParticleFilter1:

    def __init__(self, n_particles, imu_segment, uwb_ref, uwb_range):
        self.n_particles = n_particles
        self.seg = imu_segment
        self.pose = imu_segment[-1]
        self.ref = uwb_ref
        self.range = uwb_range
        self.particles = [State(0,0,0) for i in range(0,n_particles)]
        self.weights = [0 for i in range(0, n_particles)]

    def generate(self): # Why not drawing anything?
        # Generate points in a circle around pose with the length of the segment
        start_pos, start_or = State2Vec(self.seg[0])
        end_pos, end_or = State2Vec(self.pose)

        seg_len = norm(end_pos - start_pos) # circle radius
        area = 2 * seg_len * np.pi

        # This should be calculated out of VO angular velocity not out of the integrated change!!!
        # TODO: How do we add uniform noise (within a range that scales with segment curvature) to the orientation of each particle

        # Evenly spread n_particles (x,y)s over the interior of this circle
        # Just draw from uniform distribution
        for i in range(0, self.n_particles):
            r = random.uniform(0, seg_len)
            theta = random.uniform(0, 2*np.pi)

            pos_x = self.pose.x + r * np.cos(theta)
            pos_y = self.pose.y + r * np.sin(theta)
            # Direction of our current pose + some variance based on segment curvature
            ori = self.pose.o + random.uniform(-np.pi/2 , +np.pi/2) # For now we will just add +- 45deg to wherever the last pose is pointing
            # TODO: THe range that we draw in uniform should scale with the distribution of angular changes along the segment 
            self.particles[i] = State(pos_x, pos_y, ori)
            self.weights[i] = 1/self.n_particles # Because we're creating them from uniform distribution

        for part in self.particles:
            dparticle(part, color='purple')

        return None
    
    def generate_gaussian(self):
        #    particles[:, 0] = mean[0] + (randn(N) * std[0])

        # Generate points in a gaussian circle around pose with the length of the segment
        start_pos, start_or = State2Vec(self.seg[0])
        end_pos, end_or = State2Vec(self.pose)

        seg_len = norm(end_pos - start_pos) # circle radius
        area = 2 * seg_len * np.pi

        # From Gaussian distribution
        for i in range(0, self.n_particles):
            r = random.uniform(0, seg_len)
            theta = random.uniform(0, 2*np.pi)

            pos_x = self.pose.x + r * np.cos(theta)
            pos_y = self.pose.y + r * np.sin(theta)
            # Direction of our current pose + some variance based on segment curvature
            ori = self.pose.o + random.uniform(-np.pi/2 , +np.pi/2) # For now we will just add +- 45deg to wherever the last pose is pointing
            # TODO: THe range that we draw in uniform should scale with the distribution of angular changes along the segment 
            self.particles[i] = State(pos_x, pos_y, ori)
            self.weights[i] = 1/self.n_particles # Because we're creating them from uniform distribution

        for part in self.particles:
            dparticle(part, color='purple')

        return None
    
    # def motion(self):
    #     return None
    
    def measurement(self):
        # Only keep points that are within uwb_range+-10cm of ref
        for particle in self.particles:
            pos, o = State2Vec(particle)
            dist_from_ref = norm(pos - self.ref)
            inner = self.range - UNCERTAIN
            outer = self.range + UNCERTAIN

            # if dist_from_ref < outer and dist_from_ref > inner:

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
    start_pose = all_gt_pose[robot_id][0] #starts at ground truth
    imu_segment = [ State(0,0,0) for i in range(0, range_T)] # An array of States
    imu_segment[0] = State(start_pose.x, start_pose.y, start_pose.orientation)

    # dbg_view = sim_time
    # dbg_view = 120*100 
    dbg_view = 61 # range_T = 30

    sum_delta_angle = 0

    estimated_poses = []


    for t in range(1,dbg_view):
        # If its time for some client in our cluster to get slammed
        # if t % SLAM_T == 0:
        #     #TODO: Do I need to add to imu_segment here also?
        #     estimated_poses.append(all_gt_pose[robot_id][t])
        if t % range_T == 0:
            # Use Particle filter here!

            true_pos = np.array([ all_gt_pose[robot_id][t].x, all_gt_pose[robot_id][t].y ])
            v_uwb = true_pos - ref_pos

            pf = ParticleFilter1(100, imu_segment, ref_pos, norm(v_uwb))
            pf.generate()

            new_seg = pf.get_estimated_segment()
            # estimated_poses.append(pf.get_estimated_segment())
            sum_delta_angle = 0
            # Reset imu segment TODO: Fix start point later, this is stacking tip-tail atop each other
            imu_segment = [ State(0,0,0) for i in range(0, range_T)] # An array of States
            imu_segment[0] = State(new_seg[-1].x, new_seg[-1].y, new_seg[-1].o)

        else:
            # Otherwise perform regular imu integration
            i = t%range_T
            vo = all_mes_vo[robot_id][t]
            prev_pose = imu_segment[i-1]

            dy = vo.fv * dT * math.sin(prev_pose.o)         # sin = O/H
            dx = vo.fv * dT * math.cos(prev_pose.o)        # cos = A/H
            do = (vo.av) * dT
            cur_pose = State(prev_pose.x + dx, prev_pose.y + dy, prev_pose.o + do)
            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is
            imu_segment[i] = cur_pose

    x, y = ([p.x for p in estimated_poses[:dbg_view]] , [p.y for p in estimated_poses[:dbg_view]])
    plt.scatter(x, y, c='blue', s=1)

    x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_view]] , [p.y for p in all_gt_pose[robot_id][:dbg_view]])
    plt.scatter(x, y, c='green', s=1)

    # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # plt.scatter(x, y, c='green', s=1)
    x, y = ([p.x for p in mes_pose[robot_id][:dbg_view]] , [p.y for p in mes_pose[robot_id][:dbg_view]])
    plt.scatter(x, y, c='red', s=1)

    plt.show()

    return estimated_poses

