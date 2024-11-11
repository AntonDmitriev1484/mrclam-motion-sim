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
        dp((particle.x, particle.y), color)
        d = 0.008 # length we want to show the direction vector at
        start, _ = State2Vec(particle)
        end = np.array( (particle.x + d*np.cos(particle.o), particle.y + d*np.sin(particle.o) )  )
        dv( start, end, color)
    else: return None

def dparticle_weights(particles):
    weights_colors = [p.weight for p in particles]
    xs, ys = [p.x for p in particles] , [p.y for p in particles]
    plt.scatter(xs, ys, c=weights_colors, cmap='plasma', s=10)

def State2Vec(state): # Returns <x,y>, orientation
    return np.array((state.x, state.y)), state.o

def Vec2State(state):
    return State()


# Mutable named tuples, require evo-slam environment python 3.7!
@dataclass
class Particle:
    x: float
    y: float
    o: float
    weight: float
    
def Particle2StateVec(particle):
    return np.array((particle.x, particle.y, particle.o))

# Reference: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
class ParticleFilter1:
    def __init__(self, n_particles):
        self.pose = None
        self.n_particles = n_particles
        self.particles = [] # TODO: Maybe do a helper for getting weights out of particles and updating them.
        
        self.fig, self.ax = plt.sublplots()
        self.ani = FuncAnimation(self.fig, self.ani_update, interval=5, blit=True)

    # # Run after update and measurement step, 
    # def ani_update(self, i):
    #     self.scat.set_offsets()

    #     return self.scat,

    def r_weights(self): # Return copy of weights in our particles array
        return [p.weight for p in self.particles]

    def generate(self, start_pose):
        # Radius of x,y s in our Gaussian multivariate
        r_dist = 0.25 # maximum distance of a particlef from start is 1/4meter in any direction
        max_turn = np.pi / 10
        states = np.zeros((self.n_particles, 3)) # nparticles rows, 3 cols

        # Create states at uniform, and weight them according to 3D Gaussian
        for i in range(self.n_particles):
            r = random.uniform(0, r_dist)
            theta = random.uniform(0, 2*np.pi)
            states[i][0] = start_pose.x + r * np.cos(theta)
            states[i][1] = start_pose.y + r * np.sin(theta)
            states[i][2] = start_pose.orientation + random.uniform(-max_turn , +max_turn)

        var_x = np.var(states[:,0], axis=0)
        var_y = np.var(states[:,1], axis=0)
        var_o = np.var(states[:,2], axis=0)
        mean_state = np.mean(states, axis=0)
        variances = np.array([var_x, var_y, var_o])
        covariance = np.diag(variances) # Create a cov matrix assuming no cross-correlations

        weights = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            particle = Particle(states[i,0],states[i,1],states[i,2],0)
            self.particles.append(particle)
            weights[i] =  multivariate_normal.pdf(Particle2StateVec(particle), mean=mean_state, cov=covariance)

        # Distribution this gives us is correct, we just need to re-normalize everything to be out of 1
        norm_weights = weights / np.sum(weights)
        for i, w in enumerate(norm_weights):
            self.particles[i].weight = w

        # Draw particles with color representing weights
        # for p in self.particles:
        #     dparticle(p)
        # dparticle_weights(self.particles)
        
    def update(self, vo):
        # Update each particle according to visual odometry measurement
        dT = 1/T
        for i in range(self.n_particles):
            dy = vo.fv * dT * math.sin(self.particles[i].o)         # sin = O/H
            dx = vo.fv * dT * math.cos(self.particles[i].o)        # cos = A/H
            do = (vo.av) * dT
            self.particles[i].x += dx
            self.particles[i].y += dy
            self.particles[i].o += do

    # Assuming UWB ref is an np vector
    def measurement(self, uwb_ref, uwb_range, hint_pos, seg_curve):

        # Only keep points that are within uwb_range+-10cm of ref
        for i in range(self.n_particles):
            pos = np.array([self.particles[i].x, self.particles[i].y])
            dist_from_ref = norm(pos - uwb_ref)
            inner = uwb_range - UNCERTAIN
            outer = uwb_range + UNCERTAIN
            # Push all weights to 0 outside of the ranging
            if not (dist_from_ref < outer and dist_from_ref > inner):
                self.particles[i].weight = 0
        # Now normalize s.t. all weights sum to 1
        total_weight = np.sum(self.r_weights())
        for i in range(self.n_particles):
            self.particles[i].weight /= total_weight

    def show_particles(self):
        dparticle_weights(self.particles)
        plt.show()

    # Select the most likely particle as a weighted average of all remaining particles
    def estimate(self):
        states = [Particle2StateVec(p) for p in self.particles]
        weights = self.r_weights()
        self.pose = np.average(states, weights = weights, axis = 0)
        return self.pose
    
    def need_resample(self): # Calculate effective n to determine if we need to resample
        weights = self.r_weights()
        neff = 1. / np.sum(np.square(weights))
        # print(f" N effective particles {neff}")
        # threshold = self.n_particles/2
        threshold = 50
        return neff < threshold

    def resample(self):
        print("Re-sampling")
        weights = self.r_weights()

        zero_weights = 0
        for p in self.particles:
            if p.weight <= 0.0001 : zero_weights += 1
        # print(f" Before resampling # 0 weights {zero_weights} ")

        N = len(self.particles)

        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(N)) 

        candidates = []
        for i in range(self.n_particles):
            if i in indexes: candidates.append(self.particles[i])

        for c_weight, index in zip(candidates, indexes):
            t = random.randint(0, self.n_particles)
            while t in indexes:
                t = random.randint(0, self.n_particles)
            # Pick a non-candidate index
            self.particles[i].weight = c_weight
            
        for i in range(self.n_particles):
            self.particles[i].weight = 1./N



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

        particles_over_time.append(pf.particles)



    camera = Camera(plt.figure())
    for ps in particles_over_time[:100]:
        dparticle_weights(ps)
        camera.snap()
        # plt.clear()

    anim = camera.animate(blit=True)



    a = AnimatedScatter()
    plt.show()


    # # Estimated poses has less than all_gt_pose because its jsut the pf estimates so dbg_staert and dbg_end are out of bounds
    # x, y = ([p[0] for p in estimated_poses[:dbg_end]] , [p[1] for p in estimated_poses[:dbg_end]])
    # plt.scatter(x, y, c='blue', s=10)

    # x, y = ([p.x for p in all_gt_pose[robot_id][:dbg_end]] , [p.y for p in all_gt_pose[robot_id][:dbg_end]])
    # plt.scatter(x, y, c='green', s=1)

    # # x, y = ([p.x for p in all_gt_pose[ref_id][:dbg_view]] , [p.y for p in all_gt_pose[ref_id][:dbg_view]])
    # # plt.scatter(x, y, c='green', s=1)
    # x, y = ([p.x for p in mes_pose[robot_id][:dbg_end]] , [p.y for p in mes_pose[robot_id][:dbg_end]])
    # plt.scatter(x, y, c='red', s=1)

    # plt.show()

    return estimated_poses

