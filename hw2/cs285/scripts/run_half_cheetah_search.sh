echo "Search for optimal batch_size and learning rate for HalfCheetah.";
for batch in 10000 30000 50000
do
    for lr in 0.005 0.01 0.02
    do
        echo "Now running on batch_size=${batch}, learning_rate=${lr}.";
        name = "q4_search_b${batch}_lr${lr}_rtg_nnbaseline";
        python $1 --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $batch -lr $lr -rtg --nn_baseline --exp_name $name;
    done
done