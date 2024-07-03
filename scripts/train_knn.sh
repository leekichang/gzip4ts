# Define datasets
datasets=("mitbih_arr" "pamap2" "wifi")

# Define seeds
seeds=({1..10})

# Define other common parameters
n_shots=20

# Loop through datasets and seeds
for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        exp_name="${model}_${dataset}_seed${seed}"
        # CUDA_VISIBLE_DEVICES=3 python trainer.py --n_shots $n_shots --dataset $dataset --exp_name $exp_name --model $model --use_tb $use_tb --seed $seed
        # echo python temp.py --exp_name "quant_fp_1004_seed${seed}" --seed $seed --n_shot 20  --k 5 --decimal 4 --dtype quant_fp --dataset $dataset 
        python temp.py --exp_name "quant_fp_1004_seed${seed}" --seed $seed --n_shot 20  --k 5 --decimal 4 --dtype quant_fp --dataset $dataset >> knn.txt
    done
done