echo "Searching for optimal batch and learning rate...";
for BATCH in 500 1000 2500 5000 7500 
do
    for LR in 0.005 0.001 0.005 0.01 0.05
    do
        echo "Now running on batch_size=${BATCH}, learning rate=${LR}."
        NAME="q2_b${BATCH}_r${LR}";
        python $1 --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b $BATCH -lr $LR -rtg --exp_name $NAME;
    done
done