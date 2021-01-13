export CUDA_VISIBLE_DEVICES=1,8,9
python ./tools/train_bdd.py --num-gpus 3 --dist-url auto --eval-only MODEL.WEIGHTS ./out/solov2_r_101_3x_multi_scale_16bs/model_0012499.pth