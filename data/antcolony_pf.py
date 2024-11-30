
import numpy as np
import matplotlib.pyplot as plt

import random
import math

from utils import *


T=100
UNCERTAIN = 0.01 # 10cm
# Constants for accessing self.particles
X, Y, O, W = (0,1,2,3)


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

        r_dist = 0.25
        max_turn = np.pi / 12

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
            
            do = (vo.av) * dT
            self.particles[i, O] += do

            dy = vo.fv * dT * math.sin(self.particles[i, O])         # sin = O/H
            dx = vo.fv * dT * math.cos(self.particles[i, O])        # cos = A/H

            self.particles[i, X] += dx
            self.particles[i, Y] += dy


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
            # self.particles[i, W] *= p_uwb 
            # *= approach may not be working when we spawn a ton of particles, multiplying by p_UWB ends up being negligible
            # relative difference?
            # But we need *= as that is what carries over probabilities from the last step
            self.particles[i, W] = p_uwb

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
    
    # def resample(self):
    #     print("Resampling")
    #     cumulative_sum = np.cumsum(self.particles[:,W])
    #     cumulative_sum[-1] = 1. # avoid round-off error
    #     indexes = np.searchsorted(cumulative_sum, np.random.rand(self.N*self.M))
    #     # Generate N random numbers. Search for the cumulative desnities spots (particles) that are closest to that random number

    #     # resample according to indexes
    #     self.particles[:] = self.particles[indexes]
    #     self.particles[:,W] = 1.0/(self.N * self.M)
    
    def resample(self):
        print("Resampling")

        thresh = 0.0001
        denom = 0

        # Bug might be in schizophrenic iterations you're doing
        # Maybe just convert to 2D array where each row is a single position and unique rotations
        # Iterate through 2D array, and just flatten the array when you need to

        # Also, this only seems to be taking out variance from the particle distribution
        # It doesn't seem like we're properly deleting the particles that are useless
        # Maybe because everything has uniform weight before we re-sample?
        # It seems measurement is updating way less particles than its supposed to
        # It's like there is one effective measurement at the start, and then weights fade out to 0
        # Even with a more correct measurement step, we still have a lot of useless particles hovering over IMU


        for i in range(0 , self.N * self.M, self.M):
            for j in range(0, self.N * self.M, self.M):

                dist = abs(norm(self.particles[i,[X,Y]] - self.particles[j,[X,Y]]))
                d_weight = abs(self.particles[i,W] - self.particles[j, W])
                denom += (dist* d_weight) * self.M

        max_p = 0
        for i in range(0 , self.N * self.M, self.M):
            for j in range(0, self.N * self.M, self.M):

                dist = abs(norm(self.particles[i,[X,Y]] - self.particles[j,[X,Y]]))
                d_weight = abs(self.particles[i,W] - self.particles[j, W])
                
                p_move = (dist * d_weight) / denom
                if p_move > max_p: max_p = p_move

                # If our probability is above the threshold, move all particles with the same position as i, to j
                if p_move > thresh:
                    self.particles[ i:(i+self.M),[X,Y]] = self.particles[ j:(j+self.M),[X,Y]]

        print(f" Max movement prob  = { max_p}")
        self.show_particles()
       
