export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7,8
python ./tools/train_bdd.py --num-gpus 8 --dist-url tcp://0.0.0.0:12340