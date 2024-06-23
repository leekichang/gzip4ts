python temp.py --exp_name test --n_shot 20  --k 5  --decimal 2  --dtype quant_fp_novnone --model knn --dataset pamap2
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 2  --dtype quant_fp --model knn --dataset pamap2
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 2  --dtype quant_fp_novhybrid --model knn --dataset pamap2
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 2  --dtype quant_fp_novchannelwise --model knn --dataset pamap2
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 2  --dtype quant_fp_novround --model knn --dataset pamap2

python temp.py --exp_name test --n_shot 20  --k 5  --decimal 4  --dtype quant_fp_novnone --model knn --dataset mitbih_arr
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 4  --dtype quant_fp --model knn --dataset mitbih_arr
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 4  --dtype quant_fp_novhybrid --model knn --dataset mitbih_arr
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 4  --dtype quant_fp_novchannelwise --model knn --dataset mitbih_arr
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 4  --dtype quant_fp_novround --model knn --dataset mitbih_arr

python temp.py --exp_name test --n_shot 20  --k 5  --decimal 1  --dtype quant_fp_novnone --model knn --dataset wifi
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 1  --dtype quant_fp --model knn --dataset wifi
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 1  --dtype quant_fp_novhybrid --model knn --dataset wifi
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 1  --dtype quant_fp_novchannelwise --model knn --dataset wifi
python temp.py --exp_name test --n_shot 20  --k 5  --decimal 1  --dtype quant_fp_novround --model knn --dataset wifi

