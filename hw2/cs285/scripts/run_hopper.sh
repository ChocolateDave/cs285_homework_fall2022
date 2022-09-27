echo "Search for the optimal GAE lambda setting...";
for LAMBDA in 0.00 0.95 0.98 0.99 1.00
do
    echo "Now running on gae_lambda = ${LAMBDA}.";
    NAME="q5_b2000_r0.001_lambda${LAMBDA}";
    python $1 --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda $LAMBDA --exp_name $NAME;
done