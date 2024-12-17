import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import random
import math

from utils import *


T=100
UNCERTAIN = 0.01 # 10cm
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

class ParticleFilter2:
    def __init__(self, n_particles):
        self.pose = None
        self.N = n_particles
        self.particles = np.zeros((n_particles, 4)) # n_particles rows, 4 columns
        self.particles_over_t = []
        self.max_tail = 0.75
        self.furthest_on_tail = 0

    def norm_particles(self):
        self.particles[:,W] = self.particles[:,W] / np.sum(self.particles[:,W])

    def generate(self, start_pose):
        # Radius of x,y s in our Gaussian multivariate
        r_dist = 0.2 # maximum distance of a particlef from start is 1/4meter in any direction
        # Slightly less hallucination with 0.5
        max_turn = np.pi / 12

        # Create states at uniform, and weight them according to 3D Gaussian
        for i in range(self.N):
            r = random.uniform(0, r_dist)
            theta = random.uniform(0, 2*np.pi)
            self.particles[i,X] = start_pose.x + r * np.cos(theta)
            self.particles[i,Y] = start_pose.y + r * np.sin(theta)
            self.particles[i,O] = start_pose.orientation + random.uniform(-max_turn , +max_turn)

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

            # dist_on_tail = norm(self.particles[i,[X,Y]] - self.pose[[X,Y]])
            # if dist_on_tail > self.furthest_on_tail: self.furthest_on_tail = dist_on_tail


    # Virtual particle re-sampling
    def measurement(self, uwb_ref, uwb_range, seg_curvature):
        TURN_CEIL = 0.10745999999999996
        curve_ratio = (seg_curvature/TURN_CEIL)

        UWB_ERROR = 0.1 # Error is 10cm
        B = 1
        noise_limit = 0.1

        print(f"B {B} noise_limit {noise_limit}")
        # Do we want to change B or noise limit?
        
        sum_particle_weight = np.sum(self.particles[:,W])
        particles_replaced_count = 0
        self.norm_particles()

        # Pre-integrating our normal pdf for faster cdf lookup times 
        get_p_uwb = build_p_uwb_func(uwb_range, UWB_ERROR)

        if (self.pose is None): self.estimate()
        mean = self.pose[[X,Y]]
        v_uwb = mean - uwb_ref

        
        # dv(uwb_ref, uwb_ref + v_uwb)

        for i in range(self.N):
            Vparticles = np.zeros((B, self.particles.shape[1]))
            Vparticles[:] = self.particles[i]
            norm_weight = self.particles[i,W] / sum_particle_weight
            # noise = noise_func(norm_weight)
            # noise = noise_limit * (1 - norm_weight) # Default noise function
            # Vparticles = perturb(Vparticles, noise, noise)

            pos = self.particles[i,[X,Y]]
            dist_from_ref = norm(pos - uwb_ref) # Now just check how this distance falls on our UWB distribution

            p_uwb = get_p_uwb(dist_from_ref)
            self.particles[i, W] = p_uwb

            v_dev = mean - pos
            alpha = abs(norm(dot(v_dev,v_uwb))) 
            # Measure orthogonality of particle deviation direction to ultrawideband direction

            unit_v_uwb = unit(v_uwb)
            theta_rotation = rad_between_vec([1,0], unit_v_uwb)
            R_g_to_uwb = rotation_matrix(theta_rotation)

            # v_dev_uwb = dot(R_g_to_uwb, v_dev)
            v_dev_uwb = dot(v_dev, R_g_to_uwb)

            sigma_uwb_x = [0, 0]
            sigma_uwb_y = sorted([ v_dev_uwb[Y] * (0.8), v_dev_uwb[Y] * 1])

            for j in range(B):
                pos = Vparticles[j,[X,Y]]

                pos_uwb = dot(R_g_to_uwb, pos)

                # Perturbed position in UWB frame
                x_uwb_perturb = random.uniform(sigma_uwb_x[0], sigma_uwb_x[1])
                y_uwb_perturb = random.uniform(sigma_uwb_y[0], sigma_uwb_y[1])
                perturbation_vector = np.array((x_uwb_perturb, y_uwb_perturb))
                Ppos_uwb = pos_uwb + perturbation_vector
                
                R_uwb_to_g = R_g_to_uwb.T
                # Ppos = dot(R_uwb_to_g, Ppos_uwb) # Perturbed position in global frame
                Ppos = dot(Ppos_uwb, R_uwb_to_g)
                # dv(pos, Ppos)

                dist_from_ref = norm(Ppos - uwb_ref) # Now just check how this distance falls on our UWB distribution
                
                p_uwb = get_p_uwb(dist_from_ref)
                Vparticles[j, W] = p_uwb
                # If the weight of our virtual particle is greater, replace our original with it
                # if Vparticles[j,W] > (self.particles[i,W] * alpha): 
                #     self.particles[i,W] = Vparticles[j,W]
                #     particles_replaced_count+=1
                self.particles[i,W] = Vparticles[j,W]
                particles_replaced_count+=1

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
    
    def need_crop(self):
        too_wide = self.furthest_on_tail > self.max_tail
        return too_wide
    
    def cropping_resample(self):
        self.furthest_on_tail = 0

    
    def need_resample(self, seg_curvature): # Calculate effective n to determine if we need to resample
        self.norm_particles()
        TURN_CEIL = 0.10745999999999996
        curve_ratio = (seg_curvature / TURN_CEIL)**2
    
        weights = self.particles[:,W]
        neff = 1. / np.sum(np.square(weights))
        print(f" N effective particles {neff}")
        threshold = self.N/2
        # threshold = (self.N/2) * curve_ratio
        # At a certain point we stop re-sampling?
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
        max_turn = np.pi / 12
        for i in roughen_indexes:
            self.particles[i,O] += random.uniform(-max_turn , +max_turn)


        self.show_particles()