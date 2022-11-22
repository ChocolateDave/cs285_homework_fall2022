# !/usr/bin/env bash
echo "Running Homework 5 Problem 4 sp4 - offline learning with AWAC";
for TAU in 0.5 0.6 0.7 0.8 0.9 0.95 0.99
do
# Easy Environments
python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_iql_easy_supervised_lam1_tau${TAU} --use_rnd --num_exploration_steps=20000 --awac_lambda 1 --iql_expectile $TAU -gpu_id 0 &

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_iql_easy_unsupervised_lam1_tau${TAU} --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda 1 --iql_expectile $TAU -gpu_id 0 &

# Medium Environments
python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam1_tau${TAU} --use_rnd --num_exploration_steps=20000 --awac_lambda 1 --iql_expectile $TAU -gpu_id 1 &
python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam20_tau${TAU} --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda 20 --iql_expectile $TAU -gpu_id 1 &
done
wait
echo "All done!"