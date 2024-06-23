# # Define datasets
# datasets=("keti" "motion" "wifi" "pamap2" "seizure" "mitbih_id" "mitbih_arr")

# # Define seeds
# seeds=({1..10})

# # Define other common parameters
# n_shots=20
# models=("CNN" "ResNet")
# use_tb="T"

# # Loop through datasets and seeds
# for model in "${models[@]}"; do
#     for dataset in "${datasets[@]}"; do
#         for seed in "${seeds[@]}"; do
#             exp_name="${model}_${dataset}_seed${seed}"
#             CUDA_VISIBLE_DEVICES=3 python trainer.py --n_shots $n_shots --dataset $dataset --exp_name $exp_name --model $model --use_tb $use_tb --seed $seed
#         done
#     done
# done


datasets=("pamap2")

# Define seeds
seeds=(0)

# Define other common parameters
n_shots=(1 5 10 20 40 80 16)
models=("CNN" "ResNet")
use_tb="T"

# Loop through datasets and seeds
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        for seed in "${n_shots[@]}"; do
            exp_name="${model}_${dataset}_nshot${seed}"
            CUDA_VISIBLE_DEVICES=3 python trainer.py --n_shots $seed --dataset $dataset --exp_name $exp_name --model $model --use_tb $use_tb --seed 0
        done
    done
done

