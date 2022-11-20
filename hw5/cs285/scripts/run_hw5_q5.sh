# !/usr/bin/env bash
LAMBDA= # {best lambda part 4}
echo "Running Homework 5 Problem 4 sp4 - offline learning with AWAC";
for TAU in 0.5 0.6 0.7 0.8 0.9 0.95 0.99
do
# Easy Environments
python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_supervised_lam${LAMBDA}_tau${TAU} --use_rnd --num_exploration_steps=20000 --awac_lambda $LAMBDA --iql_expectile $TAU -gpu_id $1 &

python cs285/scripts/run_hw5_iql.py --env_name PointmassEasy-v0 --exp_name q5_easy_unsupervised_lam{}_tau{} --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda $LAMBDA --iql_expectile $TAU -gpu_id $1 &

# Medium Environments
python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_supervised_lam${LAMBDA}_tau${TAU} --use_rnd --num_exploration_steps=20000 --awac_lambda $LAMBDA --iql_expectile $TAU -gpu_id $1 &
python cs285/scripts/run_hw5_iql.py --env_name PointmassMedium-v0 --exp_name q5_iql_medium_unsupervised_lam${LAMBDA}_tau${TAU} --use_rnd --unsupervised_exploration --num_exploration_steps=20000 --awac_lambda $LAMBDA --iql_expectile $TAU -gpu_id $1 &
done
wait
echo "All done!"