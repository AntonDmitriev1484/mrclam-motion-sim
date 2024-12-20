from load_data import *
from algo import *
from pf_algo import *

T=100

def crop_data(gt_pose, vo):
    # start and end times
    start_gt_pose = gt_pose[0][0]
    end_gt_pose = gt_pose[-1][0]
    start_vo = vo[0][0]
    end_vo = vo[-1][0]

    # If start_vo starts after start_gt_pose cut pose data to start at the same time as start_vo
    if start_gt_pose < start_vo:
        gt_pose = list(filter(lambda x: x[0]>start_vo , gt_pose))
    else:
        vo = list(filter(lambda x: x[0]>start_gt_pose , vo))

    # If vo ends after gt_pose  cut vo data to end at gt_pose data
    if end_gt_pose < end_vo:
        vo = list(filter(lambda x: x[0]<end_gt_pose , vo))
    else:
        gt_pose = list(filter(lambda x: x[0]<end_vo , gt_pose))
    return gt_pose, vo


all_gt_pose = [ [] for i in range(1,6) ]
all_mes_vo = [ [] for i in range(1,6) ]
all_mes_pose = [ [] for i in range(1,6) ]


for i in range(1,2):
    gt_pose_df, vo_df = load_MRCLAM(1, i)
    gt_pose_df, vo_df = crop_data(gt_pose_df, vo_df)

    gt_pose_df = normalize_time(gt_pose_df)
    gt_pose = interpolate_data(dataframe_to_posetuple(gt_pose_df), T) # Gives us pose on order of cs
    all_gt_pose[i-1] = gt_pose

    vo_df = normalize_time(vo_df)
    mes_vo = interpolate_data(dataframe_to_odometrytuple(vo_df), T) # Gives us pose on order of cs
    mes_pose = measured_vo_to_pose(gt_pose, mes_vo)
    all_mes_pose[i-1] = mes_pose
    all_mes_vo[i-1] = mes_vo

    # SLAM_T = 30 # default edgeSLAM runtime is between 200 to 300ms
    # fSLAM_pose = measured_vo_to_fakeSLAM(gt_pose, mes_pose, mes_vo, 1000, T) # Run "SLAM" every 20 cs

    # write_pose_data(f"R{i}_mes", mes_pose)
    # write_pose_data(f"R{i}_gt", gt_pose)
    # write_pose_data(f"R{i}_fSLAM", fSLAM_pose)

    debug_crop = 300*100
    write_pose_data_TUM(f"R{i}_mes", mes_pose[:debug_crop])
    # write_pose_data_TUM(f"R{i}_gt", gt_pose[:1000])
    write_pose_data_TUM(f"R{i}_gt", gt_pose[:debug_crop])
    # write_pose_data_TUM(f"R{i}_fSLAM", fSLAM_pose)

    # Originally had this set to 300 -> one slam every 3000ms
    range_T = 30 # Ranging once every 300ms - i.e one member of the cluster gets Slammed every 300ms
    # This frequency makes a big impact on how long we can track the pose with ground truth
    SLAM_T = 1500 
    estimated = run_pf2(1, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=all_mes_pose)
    estimated_pose_as_pose_tuple = [ Pose(estimated[i, 0], estimated[i, 1], estimated[i, 2], estimated[i,3]) for i in range(estimated.shape[0])]
    for j in range(5):
        print(estimated_pose_as_pose_tuple[j].time)
    
    write_pose_data_TUM(f"R{i}_pf", estimated_pose_as_pose_tuple) #TODO: Comment back in when you want to use evo






# for i in range(1,2):
#     # approx_pose =  measured_vo_to_algo1(1, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=all_mes_pose)
#     # estimated_pose = run_original_pf(1, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=all_mes_pose)
#     # estimated_pose = run_antcolony_pf(1, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=all_mes_pose)
#     # estimated_pose = run_dual_pf(1, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=all_mes_pose)
#     estimated = run_pf2(1, all_gt_pose, all_mes_vo, range_T, SLAM_T, mes_pose=all_mes_pose)
#     estimated_pose_as_pose_tuple = [ Pose(0, estimated[i, 0], estimated[i, 1], estimated[i, 2]) for i in range(estimated.shape[0])]
#     write_pose_data_TUM(f"R{i}_pf", estimated_pose_as_pose_tuple) #TODO: Comment back in when you want to use evo