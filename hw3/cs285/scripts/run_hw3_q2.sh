echo "Running Homework 3 Question 2"
python $1 --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1 -gpu_id $2
python $1 --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2 -gpu_id $2
python $1 --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3 -gpu_id $2

python $1 --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --double_q --seed 1 -gpu_id $2
python $1 --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --double_q --seed 2 -gpu_id $2
python $1 --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --double_q --seed 3 -gpu_id $2
echo "Question 2 Done!"