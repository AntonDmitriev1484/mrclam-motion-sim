# from vpython import *
from data.load_data import *
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

T = 100

#TODO: Create a Fake SLAM path for each robot based on measured_pose, measured_odometry, and ground_truth
#       Tune it to be around SLAM level accurate (6-10cm ATE)
#TODO: Your algorithm should run live in the simulation and generate another dataset for each robot based on all of their fakeSLAM and ground_truth (for ranging) data


fig, ax = plt.subplots()
ax.set_xlim(-7,7)
ax.set_ylim(-7,7)
ax.grid(True)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plots = []
colors = [(color,'black') for color in ['red','blue','green','purple','orange']] # Create 5 tuples of (color, darkcolor)
plots = [(plt.scatter([], [], c=c_pair[0], s = 1), plt.scatter([], [], c=c_pair[1], s = 1)) for c_pair in colors] # Generate plots from tuples of (color, darkcolor)

plots_one_robot = [(plt.scatter([], [], c='green', s = 10), plt.scatter([], [], c='red', s = 1), plt.scatter([], [], c='blue', s = 1))]
plots = plots_one_robot


# each file = one array of Pose structs
# gt_poses is an array of [R1_gt_pose, R2_gt_pose ...]
gt_poses = [read_pose_data(f"R{i}_gt") for i in range(1,6)]
mes_poses = [read_pose_data(f"R{i}_mes") for i in range(1,6)]
fSLAM_poses = [read_pose_data(f"R{i}_fSLAM") for i in range(1,6)]


SIM_RUNTIME = len(gt_poses[0])
print(SIM_RUNTIME)


joined_gt_poses = [ np.c_[[ row.x for row in gt_pose] , [ row.y for row in gt_pose]] for gt_pose in gt_poses]
joined_mes_poses = [ np.c_[[ row.x for row in mes_pose] , [ row.y for row in mes_pose]] for mes_pose in mes_poses]
joined_fSLAM_poses = [ np.c_[[ row.x for row in fSLAM_pose] , [ row.y for row in fSLAM_pose]] for fSLAM_pose in fSLAM_poses]
# joins = list(zip( joined_gt_poses, joined_mes_poses))
joins = list(zip( joined_gt_poses, joined_mes_poses, joined_fSLAM_poses)) # pain in the ass exists so matplotloib can display data
# GIve you 5 tuples one per robot, match with plots scheme- gt will be lighter color [ (gt[x,y] , m[x,y]) ...]


# label_text = ax.text(0, 0, 'R1 GT', fontsize=12, ha='right', va='bottom')
# time_text = ax.text(5, 5.75, 't=0', fontsize=12)

TRAIL_TIME=500 # 500 = show last 5seconds of motion
POINT_INTERVAL=100 # 500 = one point every 5s
FRAME_INTERVAL=100 # 1 = 1 frame per centisecond
DISPLAY_INTERVAL_MS= 10
DISPLAY_CALCULATED = True # T/F display the calculated data
# Animation function to update the plot
def animate(i):
    # Update every 60s
    for plot, join in zip(plots, joins):
        plot[0].set_offsets(join[0][i-TRAIL_TIME:i:POINT_INTERVAL])
        if DISPLAY_CALCULATED:
            plot[1].set_offsets(join[1][i-TRAIL_TIME:i:POINT_INTERVAL])
            plot[2].set_offsets(join[2][i-TRAIL_TIME:i:POINT_INTERVAL])
    # for j in range(0,5):
    #     plots[j][0].set_offsets(joins[j][0][0:i])
    #     # plots[j][0].set_offsets(joins[j][0][i-TRAIL_TIME:i:POINT_INTERVAL])
    #     if DISPLAY_CALCULATED:
    #         plots[j][1].set_offsets(joins[j][1][i-TRAIL_TIME:i:POINT_INTERVAL])
    #         plots[j][2].set_offsets(joins[j][2][i-TRAIL_TIME:i:POINT_INTERVAL])

    return plots[0][0], plots[0][1], plots[0][2]
#     return plots[0][0], plots[0][1], plots[1][0], plots[1][1],
# plots[2][0], plots[2][1], plots[3][0], plots[3][1], plots[4][0], plots[4][1],
# label_text, time_text,
    # return plots[0][0], plots[1][0], plots[2][0], plots[3][0], plots[4][0] # Just displaying robot motion
    # return plots[3][0], plots[3][1], plots[4][0], plots[4][1],
# label_text, time_text,
    # comma here is really important for some reason


# SO I just think it can't handle more than 4 returned plots lol, why the fuck is this the return type they need

# One frame for every one minute of data.
ani = FuncAnimation(fig, animate, frames=range(0, SIM_RUNTIME, FRAME_INTERVAL), interval=DISPLAY_INTERVAL_MS, blit=True)
plt.show()
