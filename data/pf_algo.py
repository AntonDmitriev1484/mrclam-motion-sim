import matplotlib
# matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from load_data import * 
from dataclasses import dataclass
from scipy.stats import multivariate_normal

from matplotlib.animation import FuncAnimation
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

def trig_approx(ref_pose, true_pose, imu_pose, imu_dir, seg_delta_angle, plt=None):
    UWB_range = np.linalg.norm(ref_pose - true_pose)

    v_0 = imu_pose - ref_pose
    v_UWB = UWB_range * (v_0 / np.linalg.norm(v_0))

    v_UWBR = rotate_vector(v_UWB,+90) # Check signage
    A=unit(dot(imu_dir,v_UWBR))
    B=-unit(dot(imu_dir,v_UWB))
    S=rotate_vector(imu_dir,A*B*90)
    # I think if I take this approach further it might start messing up

    # R = rotate_vector(v_UWB,+90)
    # L = rotate_vector(v_UWB,-90)
    # if dot(R, imu_dir) >= 0:
    #     S = rotate_vector(imu_dir, +90)
    # else:
    #     S=rotate_vector(imu_dir, -90)

    # dv(imu_pose, imu_pose + 10*S, color='red')
    # S = np.dot(rotate_vector(v_UWB, 90), imu_dir) / np.linalg.norm(np.dot(rotate_vector(v_UWB, 90), imu_dir))
    return S

# Constants for accessing self.particles
X, Y, O, W = (0,1,2,3)

# Reference: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
class ParticleFilter1:
    def __init__(self, n_particles):
        self.pose = None
        self.N = n_particles
        self.particles = np.zeros((n_particles, 4)) # n_particles rows, 4 columns
        self.particles_over_t = []

    def norm_particles(self):
        self.particles[:,W] = self.particles[:,W] / np.sum(self.particles[:,W])

    def generate(self, start_pose):
        # Radius of x,y s in our Gaussian multivariate
        r_dist = 0.5 # maximum distance of a particlef from start is 1/4meter in any direction
        # Slightly less hallucination with 0.5
        max_turn = np.pi / 8

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

        def normal_pdf(x, mean, std_dev):
            """Calculates the Gaussian probability density function for a given value x."""
            exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)
            return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(exponent)

        for i in range(self.N):
            pos = self.particles[i,:O]

            dist_from_ref = norm(pos - uwb_ref) # Now just check how this distance falls on our UWB distribution
            p_uwb = normal_pdf(dist_from_ref, uwb_range, UNCERTAIN)
            self.particles[i, W] *= p_uwb
        
        # Now normalize s.t. all weights sum to 1
        self.norm_particles()

        
        self.show_particles()


    def show_particles(self):
        if False:
            dparticle_weights(self.particles)
            plt.show()
            plt.clf()

    # Select the most likely particle as a weighted average of all remaining particles
    def estimate(self):
        self.pose = np.average(self.particles[:,:W] , weights = self.particles[:,W], axis = 0)
        return self.pose
    
    def need_resample(self, seg_curvature): # Calculate effective n to determine if we need to resample
        self.norm_particles()
        TURN_CEIL = 0.10745999999999996
        curve_ratio = (seg_curvature / TURN_CEIL)**2
    
        weights = self.particles[:,W]
        neff = 1. / np.sum(np.square(weights))
        # print(f" N effective particles {neff}")
        threshold = self.N/2
        # At a certain point we stop re-sampling?
        return neff < 100
    
    def resample(self, seg_curvature, hint):
        # Maybe this PARTICLE_FLOOR should scale with segment curvature
        # add more noise the more curvature there is because we are less certain of our answer <- this strat works better it seems
        # or add more noise with less curvature, so we can recover closer to the GT line.

        print("Resampling")
        # N_noise = np.sum(self.particles[:,W]) / self.N # Giving very small -> 0.0005 results
        # Why are our weights driven so low by the time we resample?
        N_noise = 1000
        ceil = 1000

        print(f"Noise particles formula: {N_noise}")
        TURN_CEIL = 0.10745999999999996
        curve_ratio = (seg_curvature/TURN_CEIL)**2

        print(f"Curve ratio {curve_ratio}")
        if curve_ratio > 0: 
            N_noise = 100 + (ceil-100)*curve_ratio # set a ceiling of 1000 and floor of 100 noise particles

        N_noise = int(N_noise)
        print(f"Noise particles after curve: {N_noise}")

        self.show_particles()

        cumulative_sum = np.cumsum(self.particles[:,W])
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N))
        # Generate N random numbers. Search for the cumulative desnities spots (particles) that are closest to that random number

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.particles[:,W] = 1.0/self.N 
        
        # Once all particles weigh the same, we add noise to the data distribution
        # by randomly moving / re-orienting 100 particles.
        r_dist = 0.25

        # Lets set a range on variance between 8 and 24
        # higher curve ratio brings us closer to 8, lower to 24
        denom = 24 - (16*curve_ratio)
        max_turn = np.pi / denom

        # Generate noise particles in direction of our hint

        dv(self.pose[:O], self.pose[:O]+hint, color='blue')

        for _ in range(N_noise):
            i = random.randint(0, self.N-1) # Pick a random particle to permute
            if not (i in indexes): # If this is not one of our important points
                r = random.uniform(0, r_dist)
                theta = random.uniform(0, 2*np.pi)

                self.particles[i,X] += r * np.cos(theta)
                self.particles[i,Y] += r * np.sin(theta)
                self.particles[i,:O] += (r)*hint # All noise particles get bumped towards the direction of the GT line
                self.particles[i,O] += random.uniform(-max_turn , +max_turn)

        self.show_particles()

class AntColonyParticleFilter:
    def __init__(self, n_particles, m_particles):
        self.pose = None
        self.N = n_particles # position particles
        self.M = m_particles # orientation particles
        self.particles = np.zeros((n_particles*m_particles, 4)) # n_particles rows, 4 columns
        self.particles_over_t = []

    def norm_particles(self):
        self.particles[:,W] = self.particles[:,W] / np.sum(self.particles[:,W])

    def generate(self, start_pose):

        r_dist = 0.5 
        max_turn = np.pi / 8


        # Create states at uniform, and weight them according to 3D Gaussian
        for i in range(0, self.N*self.M, self.M):

            r = random.uniform(0, r_dist)
            theta = random.uniform(0, 2*np.pi)

            # We have N particles with random positions
            pos_x = start_pose.x + r * np.cos(theta)
            pos_y = start_pose.y + r * np.sin(theta)

            # Initialize M particles with the same position, but different orientation
            for j in range(0, self.M):
                self.particles[i+j,X] = pos_x
                self.particles[i+j,Y] = pos_y
                self.particles[i+j,O] = start_pose.orientation + random.uniform(-max_turn , +max_turn)
                # print(f"Generated particle: {self.particles[i+j, ]}")

        # Initialize all particles to have uniform weights
        for i in range(0, self.N*self.M):
            self.particles[i, W] = 1/(self.N*self.M)
        
    def update(self, vo):
        # Update each particle according to visual odometry measurement
        dT = 1/T
        for i in range(self.N*self.M):
            # Check that I am not just propogating along VO!!!
            # I should be propogating along VO + the random theta I originally generated

            dy = vo.fv * dT * math.sin(self.particles[i, O])         # sin = O/H
            dx = vo.fv * dT * math.cos(self.particles[i, O])        # cos = A/H
            do = (vo.av) * dT
            self.particles[i, X] += dx
            self.particles[i, Y] += dy
            self.particles[i, O] += do

    # Assuming UWB ref is an np vector
    def measurement(self, uwb_ref, uwb_range):
        # Only keep points that are within uwb_range+-10cm of ref

        def normal_pdf(x, mean, std_dev):
            """Calculates the Gaussian probability density function for a given value x."""
            exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)
            return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(exponent)

        for i in range(self.N*self.M):
            pos = self.particles[i,[X,Y]]

            dist_from_ref = norm(pos - uwb_ref) # Now just check how this distance falls on our UWB distribution
            p_uwb = normal_pdf(dist_from_ref, uwb_range, UNCERTAIN)
            self.particles[i, W] *= p_uwb
        
        # Now normalize s.t. all weights sum to 1
        self.norm_particles()
        print("Measurement")
        self.show_particles()


    def show_particles(self):
        if False:
            dparticle_weights(self.particles)
            plt.show()
            plt.clf()

    # Select the most likely particle as a weighted average of all remaining particles
    def estimate(self):
        self.pose = np.average(self.particles[:,:W] , weights = self.particles[:,W], axis = 0)
        return self.pose
    
    def need_resample(self): # Calculate effective n to determine if we need to resample
        self.norm_particles()
        weights = self.particles[:,W]
        neff = 1. / np.sum(np.square(weights))
        print(f"neff {neff}")
        return neff < 100
    
    def resample(self):
        print("Resampling")
        cumulative_sum = np.cumsum(self.particles[:,W])
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N*self.M))
        # Generate N random numbers. Search for the cumulative desnities spots (particles) that are closest to that random number

        # resample according to indexes
        self.particles[:] = self.particles[indexes]
        self.particles[:,W] = 1.0/(self.N * self.M)
    
    # def resample(self):
    #     print("Resampling")

    #     thresh = 0.25
    #     denom = 0

    #     for i in range(self.N * self.M):
    #         for j in range(self.N * self.M):
    #             dist = abs(norm(self.particles[i,[X,Y]] - self.particles[j,[X,Y]]))
    #             d_weight = abs(self.particles[i,W] - self.particles[j, W])
    #             denom += (dist* d_weight)

    #     for i in range(self.N * self.M):
    #         for j in range(self.N * self.M):
    #             dist = abs(norm(self.particles[i,[X,Y]] - self.particles[j,[X,Y]]))
    #             d_weight = abs(self.particles[i,W] - self.particles[j, W])
                
    #             p_move = (dist * d_weight) / denom

    #             # If our probability is above the threshold, move i to j
    #             if p_move > thresh:
    #                 self.particles[i,[X,Y]] = self.particles[j,[X,Y]]

    #     self.show_particles()
       


def measured_vo_to_algo2(robot_id, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=None):
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

    pf = AntColonyParticleFilter(100, 100)
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

