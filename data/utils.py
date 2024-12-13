import numpy as np
import matplotlib.pyplot as plt

T=100
UNCERTAIN = 0.01 # 10cm
X, Y, O, W = (0,1,2,3)

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

    
def normal_pdf(x, mean, std_dev):
    """Calculates the Gaussian probability density function for a given value x."""
    exponent = -((x - mean) ** 2) / (2 * std_dev ** 2)
    return (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(exponent)


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