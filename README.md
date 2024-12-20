# mrclam-motion-sim

Current development branch is 'vparticle-filter'

For running a filter, pick the appropriate 'run_pf' method in pf_algo.py
Then call that method providing odometry measurements and gt measurements in process_data.py

Pyplot runs in each 'run_pf' method for quick visualization.

Program entrypoint is executing 'process_data.py', which will display debugging pyplots and also write computed trajectory data to a file.