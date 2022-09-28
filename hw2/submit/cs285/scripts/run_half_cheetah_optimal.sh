BATCH=$2
LR=$3

echo "Running with optimal batch_size and learning rate for HalfCheetah.";
echo "Running baseline";
python $1 --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH -lr $LR --exp_name q4_b${BATCH}_r${LR};
echo "Running reward-to-go";
python $1 --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH -lr $LR -rtg --exp_name q4_b${BATCH}_r${LR}_rtg;
echo "Running with baseline neural network";
python $1 --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH -lr $LR --nn_baseline --exp_name q4_b${BATCH}_r${LR}_nnbaseline;
echo "Running with all";
python $1 --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $BATCH -lr $LR -rtg --nn_baseline --exp_name q4_b${BATCH}_r${LR}_rtg_nnbaseline