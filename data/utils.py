import numpy as np
import scipy.stats as scp

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

from scipy import integrate, interpolate

T=100
UNCERTAIN = 0.01 # 10cm
X, Y, O, W = (0,1,2,3)

def rotate_segment(imu_segment):
    return None

def rotation_matrix(theta):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    return rotation_matrix

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
    
def normal_cdf(x_l, x_u, mean, std_dev):
    p_u = scp.norm.cdf(x_u, loc=mean, scale=std_dev)
    p_l = scp.norm.cdf(x_l, loc=mean, scale=std_dev)
    # Super slow, might just pull a formula off the internet?
    return p_u - p_l

def normal_pdf(x, mean, std_dev):
    """Calculates the Gaussian probability density function for a given value x."""
    exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(exponent)

DBG = True
def dv(start, end, label=None, color='000000'):
    end2 = end-start
    if DBG: plt.arrow(start[0], start[1], end2[0], end2[1], color=color, label=label, head_width=0.001, head_length=0.001, width=0.0001)

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

def perturb(particles, x_lim, y_lim):
    particles[:,X] += random.uniform(-x_lim, x_lim)
    particles[:,Y] += random.uniform(-y_lim, y_lim)
    # var_o = np.pi/12
    # particles[:,O] += random.uniform(-var_o, var_o) # Adding in a pinch of orientation variance does help
    return particles

# center_weight = 1
# center_weight *= curve_ratio
# # More likely to drift on a harder curve, so we add more searching power to our low weight particles
# def custom_noise_func(w):
#     if center_weight == 0: return 0.05
#     left_bound = 0
#     right_bound = 1
#     m_left = noise_limit / (0+center_weight)
#     m_right = - noise_limit / (1-center_weight)
#     if (w >= center_weight):
#         return noise_limit + m_right*(w-center_weight)
#     if (w < center_weight):
#         return m_left * (w)
#     # More curve, means add more variance further out
#     # Less curve means add variance further in

def build_p_uwb_func(uwb_range, UWB_ERROR):
    n_samples = 4000
    x_values = np.linspace(uwb_range-1, uwb_range+1,n_samples)
    pdf = np.zeros(n_samples)
    for i, distance in enumerate(x_values):
        pdf[i] = normal_pdf(distance, uwb_range, UWB_ERROR)
    cdf = integrate.cumulative_trapezoid(pdf, x_values, initial=0) # Integrate our pdf over the x_values
    queryable_cdf = interpolate.interp1d(x_values, cdf, kind='linear', bounds_error=False, fill_value=0)
    
    def get_p_uwb(dist_from_ref): # Function to query our CDF
        p_mass_upper = queryable_cdf(dist_from_ref+0.01)
        p_mass_lower = queryable_cdf(dist_from_ref-0.01)
        return p_mass_upper - p_mass_lower
    
    return get_p_uwb