export CUDA_VISIBLE_DEVICES=3
python ./tools/train_bdd.py --num-gpus 1 --dist-url tcp://0.0.0.0:12340