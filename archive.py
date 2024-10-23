
# # Optional: set up the camera position
# scene.camera.pos = vector(0,0,5)
# # scene.camera.axis = vector(-2,-2,-2)

# # box(pos=vector(r1_m_pose[0].x,r1_m_pose[0].y,0), size=vector(1,1,1), color=color.green)

# trail_interval = T
# trail_radius =0.05
# box_size = vector(0.5, 0.5, 0.5)

# # # Create a 3D cube
# gt_cube = box(make_trail=True, trail_type="points", trail_radius=trail_radius, interval=trail_interval, 
#               pos=vector(r1_gt_pose[0].x,r1_gt_pose[0].y,0), size=box_size, color=color.blue)
# mv_cube = box(make_trail=True, trail_type="points", trail_radius=trail_radius, interval=trail_interval,
#               pos=vector(r1_m_pose[0].x,r1_m_pose[0].y,0), size=box_size, color=color.red)

# i = 0
# stop = 300*T
# # Note if you up T then you also need to skip
# for mov, gt in zip(r1_m_pose, r1_gt_pose):
#     rate(T*100)
#     gt_cube.pos = vector(gt.x, gt.y, 0)
#     mv_cube.pos = vector(mov.x, mov.y, 0)
#     i+=1

# while True:
#     pass