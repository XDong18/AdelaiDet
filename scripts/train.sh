export CUDA_VISIBLE_DEVICES=1,2,3,4
python ./tools/train_bdd.py --num-gpus 4 --dist-url tcp://0.0.0.0:12340