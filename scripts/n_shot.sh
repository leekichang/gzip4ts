# python mlClassifier.py --dataset pamap2 --model RF --n_shot 1   --seed 0 --exp_name n_shot_RF_1
# python mlClassifier.py --dataset pamap2 --model RF --n_shot 5   --seed 0 --exp_name n_shot_RF_5
# python mlClassifier.py --dataset pamap2 --model RF --n_shot 10  --seed 0 --exp_name n_shot_RF_10
# python mlClassifier.py --dataset pamap2 --model RF --n_shot 20  --seed 0 --exp_name n_shot_RF_20
# python mlClassifier.py --dataset pamap2 --model RF --n_shot 40  --seed 0 --exp_name n_shot_RF_40
# python mlClassifier.py --dataset pamap2 --model RF --n_shot 80  --seed 0 --exp_name n_shot_RF_80
# python mlClassifier.py --dataset pamap2 --model RF --n_shot 160 --seed 0 --exp_name n_shot_RF_160

# python mlClassifier.py --dataset pamap2 --model AB --n_shot 1   --seed 0 --exp_name n_shot_AB_1
# python mlClassifier.py --dataset pamap2 --model AB --n_shot 5   --seed 0 --exp_name n_shot_AB_5
# python mlClassifier.py --dataset pamap2 --model AB --n_shot 10  --seed 0 --exp_name n_shot_AB_10
# python mlClassifier.py --dataset pamap2 --model AB --n_shot 20  --seed 0 --exp_name n_shot_AB_20
# python mlClassifier.py --dataset pamap2 --model AB --n_shot 40  --seed 0 --exp_name n_shot_AB_40
# python mlClassifier.py --dataset pamap2 --model AB --n_shot 80  --seed 0 --exp_name n_shot_AB_80
# python mlClassifier.py --dataset pamap2 --model AB --n_shot 160 --seed 0 --exp_name n_shot_AB_160

python temp.py --exp_name quant_fp_6206 --dataset pamap2 --seed 6206 --n_shot 1    --k 5 --decimal 4 --dtype quant_fp --model gzip --seed 0
python temp.py --exp_name quant_fp_6206 --dataset pamap2 --seed 6206 --n_shot 5    --k 5 --decimal 4 --dtype quant_fp --model gzip --seed 0
python temp.py --exp_name quant_fp_6206 --dataset pamap2 --seed 6206 --n_shot 10   --k 5 --decimal 4 --dtype quant_fp --model gzip --seed 0
python temp.py --exp_name quant_fp_6206 --dataset pamap2 --seed 6206 --n_shot 20   --k 5 --decimal 4 --dtype quant_fp --model gzip --seed 0
python temp.py --exp_name quant_fp_6206 --dataset pamap2 --seed 6206 --n_shot 40   --k 5 --decimal 4 --dtype quant_fp --model gzip --seed 0
python temp.py --exp_name quant_fp_6206 --dataset pamap2 --seed 6206 --n_shot 80   --k 5 --decimal 4 --dtype quant_fp --model gzip --seed 0
python temp.py --exp_name quant_fp_6206 --dataset pamap2 --seed 6206 --n_shot 160  --k 5 --decimal 4 --dtype quant_fp --model gzip --seed 0