import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

import random
import math

from utils import *


T=100
UNCERTAIN = 0.01 # 10cm
# Constants for accessing self.particles
X, Y, O, W = (0,1,2,3)

class ParticleFilter:
    def __init__(self, n_particles):
        self.pose = None
        self.N = n_particles
        self.particles = np.zeros((n_particles, 4)) # n_particles rows, 4 columns
        self.particles_over_t = []

    def norm_particles(self):
        self.particles[:,W] = self.particles[:,W] / np.sum(self.particles[:,W])

    def generate(self, start_pose):
        r_dist = 0.25
        max_turn = np.pi / 12

        # Create states at uniform, and weight them according to 3D Gaussian
        for i in range(self.N):
            r = random.uniform(0, r_dist)
            theta = random.uniform(0, 2*np.pi)
            self.particles[i,X] = start_pose[X] + r * np.cos(theta)
            self.particles[i,Y] = start_pose[Y] + r * np.sin(theta)
            self.particles[i,O] = start_pose[O] + random.uniform(-max_turn , +max_turn)

        self.particles[:, W] = 1/self.N
        self.norm_particles()
        
    def update(self, vo):
        # Update each particle according to robot odometry measurement
        dT = 1/T
        for i in range(self.N):
            do = (vo.av) * dT
            self.particles[i, O] += do
            dy = vo.fv * dT * math.sin(self.particles[i, O])         # sin = O/H
            dx = vo.fv * dT * math.cos(self.particles[i, O])        # cos = A/H
            self.particles[i, X] += dx
            self.particles[i, Y] += dy

    # Virtual particle re-sampling
    def measurement(self, true_pos, anchors, AoA_precision, GT_orientation):

        UWB_ERROR = 0.1 # Error is 10cm
        B = 2
        noise_limit = 0.1

        particles_replaced_count = 0
        self.norm_particles()


        uwb_ref = anchors[0]
        v_uwb = true_pos - uwb_ref
        uwb_range = norm(v_uwb)


        # Pre-integrating our normal pdf for faster cdf lookup times 
        get_p_uwb = build_p_uwb_func(uwb_range, UWB_ERROR)

        particles_out_of_AoA_bounds = 0

        AoA_upper, AoA_lower = (( GT_orientation + AoA_precision/2), (GT_orientation - AoA_precision/2))
        
        # Still don't know if this is correct
        # but Ima roll with it for now
        def in_range(lower, upper, angle):
            lower %= 2*np.pi
            upper %= 2*np.pi
            angle %= 2*np.pi
            if lower > upper:
                return angle >= lower or angle <= upper
            return angle > lower and angle < upper

        for i in range(self.N):
            pos = self.particles[i,[X,Y]]
            dist_from_ref = norm(pos - uwb_ref) # Now just check how this distance falls on our UWB distribution
            p_uwb = get_p_uwb(dist_from_ref)
            self.particles[i, W] = p_uwb

            if not in_range(AoA_lower, AoA_upper, self.particles[i,[O]]):
                # print(f"{self.particles[i, [O]]}")
                # self.particles[i,W] = 0
                self.particles[i,O] = random.uniform(AoA_lower, AoA_upper)
                particles_out_of_AoA_bounds += 1

        print(f" GT: {GT_orientation} - upper: {AoA_upper} lower: {AoA_lower}")
        print(particles_out_of_AoA_bounds)

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
        # threshold = self.N/2
        threshold = (self.N/2) * curve_ratio
        return neff < threshold
    
    def resample(self):
        print(f"Resampling")
        cumulative_sum = np.cumsum(self.particles[:,W])
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N))
        # Generate N random numbers. Search for the cumulative desnities spots (particles) that are closest to that random number
        replace = self.particles[indexes]
        self.particles[:,[X,Y]] = replace[:,[X,Y]]         # resample according to indexes
        self.particles[:,W] = 1.0/self.N

        # A bit of roughening
        roughen_indexes = np.random.randint(0, self.N, (int(self.N/3)))
        self.particles[[roughen_indexes]] = perturb(self.particles[[roughen_indexes]], 0.05, 0.05)
        # max_turn = np.pi / 12
        # for i in roughen_indexes:
        #     self.particles[i,O] += random.uniform(-max_turn , +max_turn)


        self.show_particles()