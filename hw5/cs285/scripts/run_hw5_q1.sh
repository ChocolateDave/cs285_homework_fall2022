# !/usr/bin/env bash
echo "Running homework 5 Problem 1";
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd -gpu_id $1 &
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random -gpu_id $1 &
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --exp_name q1_env2_rnd -gpu_id $1 &
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --unsupervised_exploration --exp_name q1_env2_random -gpu_id $1 &
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_ucb --unsupervised_exploration --exp_name q1_alg_med -gpu_id $1 &
wait
echo "All done!"