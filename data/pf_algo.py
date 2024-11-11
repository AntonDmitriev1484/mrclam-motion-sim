import numpy as np
import matplotlib.pyplot as plt
import random
from load_data import * 
from dataclasses import dataclass
from scipy.stats import multivariate_normal
from matplotlib.animation import FuncAnimation
from celluloid import Camera
from AnimatedScatter import AnimatedScatter

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
    if DBG: return plt.scatter(pos[0], pos[1], color=color, label=label,  s=10)
    else: return None

def dparticle(particle, color='000000'):
    if DBG:
        dp((particle[X], particle[Y]), color)
        d = 0.008 # length we want to show the direction vector at
        start, _ = particle[:O] # Just want X, Y out of particle
        end = np.array( (particle[X] + d*np.cos(particle[O]), particle[Y] + d*np.sin(particle[O]) )  )
        dv( start, end, color)
    else: return None

def dparticle_weights(particles):
    weights_colors = particles[:,W]
    xs, ys = particles[:,X], particles[:,Y]
    plt.scatter(xs, ys, c=weights_colors, cmap='plasma', s=10)



# Mutable named tuples, require evo-slam environment python 3.7!
@dataclass
class Particle:
    x: float
    y: float
    o: float
    weight: float

# Constants for accessing self.particles
X, Y, O, W = (0,1,2,3)

# Reference: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
class ParticleFilter1:
    def __init__(self, n_particles):
        self.pose = None
        self.N = n_particles
        self.particles = np.zeros((n_particles, 4)) # n_particles rows, 4 columns
        
        self.fig, self.ax = plt.subplots()
        self.scat = self.ax.scatter(self.particles[:,X], self.particles[:,Y], cmap="plasma", s=10)
        
        self.ax.axis([-10, 10, -10, 10])
        self.ani = FuncAnimation(self.fig, self.ani_update, interval=1000, blit=True)

    # Run after update and measurement step, 
    def ani_update(self, i):
        self.scat.set_offsets(np.c_[ self.particles[:,X] , self.particles[:,Y]])
        self.scat.set_array(self.particles[:, W])
        return self.scat,


    def norm_particles(self):
        self.particles[:,W] = self.particles[:,W] / np.sum(self.particles[:,W])

    def generate(self, start_pose):
        # Radius of x,y s in our Gaussian multivariate
        r_dist = 0.25 # maximum distance of a particlef from start is 1/4meter in any direction
        max_turn = np.pi / 10

        # Create states at uniform, and weight them according to 3D Gaussian
        for i in range(self.N):
            r = random.uniform(0, r_dist)
            theta = random.uniform(0, 2*np.pi)
            self.particles[i,X] = start_pose.x + r * np.cos(theta)
            self.particles[i,Y] = start_pose.y + r * np.sin(theta)
            self.particles[i,O] = start_pose.orientation + random.uniform(-max_turn , +max_turn)

        var_x = np.var( self.particles[:,X], axis=0)
        var_y = np.var( self.particles[:,Y], axis=0)
        var_o = np.var( self.particles[:,O], axis=0)
        # So for NUMPY, you actually have to use the [ , ] in all scenarios for proper slicing

        mean_state = np.mean(self.particles[:,:W], axis=0)
        variances = np.array([var_x, var_y, var_o])
        covariance = np.diag(variances) # Create a cov matrix assuming no cross-correlations

        for i in range(self.N):
            self.particles[i,W] =  multivariate_normal.pdf(self.particles[i,:W], mean=mean_state, cov=covariance)

        self.norm_particles()
        # Draw particles with color representing weights
        # for p in self.particles:
        #     dparticle(p)
        # dparticle_weights(self.particles)
        
    def update(self, vo):
        # Update each particle according to visual odometry measurement
        dT = 1/T
        for i in range(self.N):
            dy = vo.fv * dT * math.sin(self.particles[i, O])         # sin = O/H
            dx = vo.fv * dT * math.cos(self.particles[i, O])        # cos = A/H
            do = (vo.av) * dT
            self.particles[i, X] += dx
            self.particles[i, Y] += dy
            self.particles[i, O] += do

    # Assuming UWB ref is an np vector
    def measurement(self, uwb_ref, uwb_range, hint_pos, seg_curve):
        # Only keep points that are within uwb_range+-10cm of ref
        for i in range(self.N):
            pos = np.array([self.particles[i, X], self.particles[i, Y]])
            dist_from_ref = norm(pos - uwb_ref)
            inner = uwb_range - UNCERTAIN
            outer = uwb_range + UNCERTAIN
            # Push all weights to 0 outside of the ranging
            if not (dist_from_ref < outer and dist_from_ref > inner):
                self.particles[i, W] = 0
        # Now normalize s.t. all weights sum to 1
        self.norm_particles()

    def show_particles(self):
        dparticle_weights(self.particles)
        plt.show()

    # Select the most likely particle as a weighted average of all remaining particles
    def estimate(self):
        self.pose = np.average(self.particles[:,:W] , weights = self.particles[:,W], axis = 0)
        return self.pose
    
    def need_resample(self): # Calculate effective n to determine if we need to resample
        weights = self.particles[:,W]
        neff = 1. / np.sum(np.square(weights))
        # print(f" N effective particles {neff}")
        # threshold = self.N/2
        threshold = 50
        return neff < threshold

    def resample(self):
        print("Re-sampling")
        weights = self.particles[:,W]

        zero_weights = 0
        for p in self.particles:
            if p[W] <= 0.0001 : zero_weights += 1
        # print(f" Before resampling # 0 weights {zero_weights} ")

        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N)) 

        candidates = []
        for i in range(self.N):
            if i in indexes: candidates.append(self.particles[i]) # Adding a whole [x,y,o,w]

        for particle in candidates:
            t = random.randint(0, self.N)
            while t in indexes:
                t = random.randint(0, self.N)
            # Pick a non-candidate index
            self.particles[i] = particle
            
        for i in range(self.N):
            self.particles[i, W] = 1./self.N



def measured_vo_to_algo2(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
    # This algorithm will use a particle filter to estimate point location during each range.

    robot_id=0
    dT = 1/T
    sim_time = min( [len(all_gt_pose[0]), len(all_gt_pose[1]) ] )

    ref_pos = np.array((0,0))
    start_pose = all_gt_pose[robot_id][0] #starts at ground truth
    imu_segment = [ State(0,0,0) for i in range(0, range_T)] # An array of States
    imu_segment[0] = State(start_pose.x, start_pose.y, start_pose.orientation)

    # Segment from 1 to 1.5 minutes has problems
    # dbg_start = 40 * 100
    dbg_start = 0
    # dbg_end = 300 * 100
    dbg_end = 120 * 100

    # dbg_view = sim_time
    # dbg_view = 120*100 
    # dbg_view = 100*range_T
    # dbg_view = 300 # range_T = 30

    sum_delta_angle = 0

    estimated_poses = []
    particles_over_time = []

    pf = ParticleFilter1(2000)
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
            estimated_poses.append(estimate)

            if pf.need_resample(): pf.resample()
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
            sum_delta_angle += abs(do) # don't care about signage, just want to capture how windy this segment is
            imu_segment[i] = cur_pose

        pf.ani_update(t)


    plt.show()

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

    return estimated_poses

