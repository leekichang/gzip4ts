# valgrind --tool=massif --suppressions=./valgrind-python.supp --pages-as-heap=yes --massif-out-file=massif.out \
# python deep_inference.py --exp_name test --model CNN --dataset mitbih_arr --n_shots 5; grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1
# valgrind --tool=massif --suppressions=./valgrind-python.supp --pages-as-heap=yes --massif-out-file=massif.out \
# python deep_inference.py --exp_name test --model ResNet --dataset mitbih_arr --n_shots 5; grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1
# valgrind --tool=massif --suppressions=./valgrind-python.supp --pages-as-heap=yes --massif-out-file=massif.out \
# python shallow_inference.py --exp_name test --model AB  --dataset mitbih_arr --n_shots 5; grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1
# valgrind --tool=massif --suppressions=./valgrind-python.supp --pages-as-heap=yes --massif-out-file=massif.out \
# python shallow_inference.py --exp_name test --model RF  --dataset mitbih_arr --n_shots 5; grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1
# valgrind --tool=massif --suppressions=./valgrind-python.supp--pages-as-heap=yes --massif-out-file=massif.out \
# python temp_inference.py --exp_name test --n_shot 5 --k 5 --decimal 2 --dtype quant_fp --model knn --dataset mitbih_arr; grep mem_heap_B massif.out | sed -e 's/mem_heap_B=\(.*\)/\1/' | sort -g | tail -n 1



python deep_inference.py --exp_name test --model CNN --dataset mitbih_arr --n_shots 5
sleep 1

python deep_inference.py --exp_name test --model ResNet --dataset mitbih_arr --n_shots 5
sleep 1

python shallow_inference.py --exp_name test --model AB  --dataset mitbih_arr --n_shots 5
sleep 1
python shallow_inference.py --exp_name test --model RF  --dataset mitbih_arr --n_shots 5
sleep 1

python temp_inference.py --exp_name test --n_shot 5 --k 5 --decimal 2 --dtype quant_fp --model knn --dataset mitbih_arr
sleep 1