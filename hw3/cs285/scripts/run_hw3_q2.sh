echo "Running Homework 3 Question 2"
python $1 --env_name LunarLander-v3 --exp_name q2_dqn_1 --seed 1
python $1 --env_name LunarLander-v3 --exp_name q2_dqn_2 --seed 2
python $1 --env_name LunarLander-v3 --exp_name q2_dqn_3 --seed 3

python $1 --env_name LunarLander-v3 --exp_name q2_doubledqn_1 --
double_q --seed 1
python $1 --env_name LunarLander-v3 --exp_name q2_doubledqn_2 --
double_q --seed 2
python $1 --env_name LunarLander-v3 --exp_name q2_doubledqn_3 --
double_q --seed 3
echo "Question 2 Done!"