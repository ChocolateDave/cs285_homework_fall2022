# !/usr/bin/env bash
echo "Running Homework 5 Problem 4 sp4 - offline learning with AWAC";
for LAMBDA in 0.1 1 2 10 20 50
do
    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --exp_name q4_awac_easy_unsupervised_lam${LAMBDA} --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda $LAMBDA -gpu_id $1 &
    python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda $LAMBDA --exp_name q4_awac_easy_supervised_lam${LAMBDA} -gpu_id $1 &
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --exp_name q4_awac_medium_unsupervised_lam${LAMBDA} --use_rnd --num_exploration_steps=20000 --unsupervised_exploration --awac_lambda $LAMBDA -gpu_id $1 &
    python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps=20000 --awac_lambda $LAMBDA --exp_name q4_awac_medium_supervised_lam${LAMBDA} -gpu_id $1 &
done
wait
echo "All done!";