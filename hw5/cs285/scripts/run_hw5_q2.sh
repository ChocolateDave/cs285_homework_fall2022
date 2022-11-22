# !/usr/bin/env bash
echo "Running Homework 5 Problem 2 sp1 - compare dqn to cql";
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.0 -gpu_id $1 &
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 -gpu_id $1 &
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_dqn_shift_scale --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.0 --exploit_rew_shift 1 --exploit_rew_scale 100 -gpu_id $1 &
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --exp_name q2_cql_shift_scale --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha=0.1 --exploit_rew_shift 1 --exploit_rew_scale 100 -gpu_id $1 &
wait
echo "SP1 done!";
echo "Running Homework 5 Problem 2 sp2 - compare size of exploration";
for STEP in 5000 15000
do
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps $STEP --offline_exploitation --cql_alpha=0.0 --unsupervised_exploration --exp_name q2_dqn_numsteps_${STEP} -gpu_id $1 &
    python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --num_exploration_steps $STEP --offline_exploitation --cql_alpha=0.1 --unsupervised_exploration --exp_name q2_cql_numsteps_${STEP} -gpu_id $1 &
done
wait
echo "SP2 done!";
echo "Running Homework 5 Problem 2 sp3 - sweep over cql alphas";
for ALPHA in 0.02 0.5
do
python cs285/scripts/run_hw5_expl.py --env_name PointmassMedium-v0 --use_rnd --unsupervised_exploration --offline_exploitation --cql_alpha $ALPHA --exp_name q2_alpha_${ALPHA} -gpu_id $1 &
done
wait
echo "SP3 done!";
echo "All done!"