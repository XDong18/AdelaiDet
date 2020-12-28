export CUDA_VISIBLE_DEVICES=5,6,7,8
python ./tools/train_bdd.py --num-gpus 4 --dist-url tcp://0.0.0.0:12345