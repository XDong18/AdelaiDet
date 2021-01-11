export CUDA_VISIBLE_DEVICES=0,1,2,3,8,9
python ./tools/train_bdd.py --num-gpus 6 --dist-url tcp://0.0.0.0:12340  --resume