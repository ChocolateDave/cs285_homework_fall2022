NTU=$2
NGSPTU=$3

echo "Running Homework 3 Question 5";
python $1 --env_name InvertedPendulum-v2 --ep_len 1000 --discount 0.95 -n 100 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name q5_${NTU}_${NGSPTU} -ntu $NTU -ngsptu $NGSPTU;
python $1 --env_name HalfCheetah-v2 --ep_len 150 --discount 0.90 --scalar_log_freq 1 -n 150 -l 2 -s 32 -b 30000 -eb 1500 -lr 0.02 --exp_name q5_${NTU}_${NGSPTU} -ntu $NTU -ngsptu $NGSPTU
echo "Done!"