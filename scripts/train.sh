export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
python ./tools/train_bdd.py --num-gpus 8 --dist-url tcp://0.0.0.0:12340  --resume