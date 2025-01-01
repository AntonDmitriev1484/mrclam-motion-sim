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
    def measurement(self, true_pos, anchors):

        UWB_ERROR = 0.1 # Error is 10cm
        B = 2
        noise_limit = 0.1
        
        sum_particle_weight = np.sum(self.particles[:,W])
        particles_replaced_count = 0
        self.norm_particles()

        pca = PCA(n_components=2)
        pca.fit(self.particles[:,[X,Y]])
        v_var = pca.components_[0]
        # Debug to make sure variance direction is correct
        if (self.pose is None): self.estimate()
        mean = self.pose[[X,Y]]
        # dv(mean, mean+ (0.1)*unit(v_var))
        # dv(mean, mean+ (0.1)*unit(v_uwb))
        
        # uwb_ref = None
        # min_dot_product = 10
        # for j in range(anchors.shape[0]):
        #     uwb_anchor = anchors[j,:]
        #     v_uwb = true_pos - uwb_anchor

        #     print(f" anchor:{uwb_anchor} -> dot: {dot( unit(v_var), unit(v_uwb))}")
        #     dot_product = abs(norm(dot( unit(v_uwb), unit(v_var)))) # Order

        #     if dot_product < min_dot_product:
        #         min_dot_product = dot_product
        #         uwb_ref = uwb_anchor

        uwb_ref = anchors[0]
        print(f" Chose anchor at {uwb_ref}")
        v_uwb = true_pos - uwb_ref
        uwb_range = norm(v_uwb)

        # Pre-integrating our normal pdf for faster cdf lookup times 
        get_p_uwb = build_p_uwb_func(uwb_range, UWB_ERROR)

        for i in range(self.N):
            v_particles = np.zeros((B, self.particles.shape[1]))
            v_particles[:] = self.particles[i]
            norm_weight = self.particles[i,W] / sum_particle_weight
            # noise = noise_func(norm_weight)
            noise = noise_limit * (1 - norm_weight) # Default noise function
            v_particles = perturb(v_particles, noise, noise)

            pos = self.particles[i,[X,Y]]
            dist_from_ref = norm(pos - uwb_ref) # Now just check how this distance falls on our UWB distribution
            p_uwb = get_p_uwb(dist_from_ref)
            self.particles[i, W] = p_uwb

            best_virtual = None
            best_weight = 0

            for j in range(B):
                pos = v_particles[j,[X,Y]]
                dist_from_ref = norm(pos - uwb_ref) # Now just check how this distance falls on our UWB distribution
                p_uwb = get_p_uwb(dist_from_ref)
                v_particles[j, W] = p_uwb

                # If the weight of our virtual particle is greater, replace our original with it
                if v_particles[j,W] > best_weight: 
                    best_weight = v_particles[j,W]
                    best_virtual = v_particles[j]
            
            if not best_virtual is None:
                if best_virtual[W] > self.particles[i,W]: 
                    self.particles[i] = best_virtual

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