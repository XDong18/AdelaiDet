export CUDA_VISIBLE_DEVICES=4,5,6,7
python ./tools/train_bdd.py --num-gpus 4 --dist-url tcp://0.0.0.0:12345