echo 'Running small batch w/o reward_to_go w/ standardized_advatages';
python $1 --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa;

echo 'Running small batch w/ reward_to_go w/ standardized_advatages'; 
python $1 --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa;

echo 'Running small batch w/ reward_to_go w/o standardized_advatages';
python $1 --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na;

echo 'Running large batch w/o reward_to_go w/ standardized_advatages';
python $1 --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa;

echo 'Running large batch w/ reward_to_go w/ standardized_advatages'; 
python $1 --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa;

echo 'Running large batch w/ reward_to_go w/o standardized_advatages';
python $1 --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na;