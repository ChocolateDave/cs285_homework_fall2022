# !/usr/bin/env bash
echo "Running homework 5 Problem 1";
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd &
python cs285/scripts/run_hw5_expl.py --env_name PointmassEasy-v0 --unsupervised_exploration --exp_name q1_env1_random &
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --use_rnd --unsupervised_exploration --exp_name q1_env1_rnd &
python cs285/scripts/run_hw5_expl.py --env_name PointmassHard-v0 --unsupervised_exploration --exp_name q1_env1_random &
wait
echo "All done!"