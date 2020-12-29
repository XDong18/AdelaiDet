export CUDA_VISIBLE_DEVICES=2,3
python ./tools/train_bdd.py --num-gpus 2 --dist-url tcp://0.0.0.0:12340  --resume ./out/solov2_r_50_3x/last_checkpoint