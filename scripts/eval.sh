export CUDA_VISIBLE_DEVICES=1,8,9
python ./tools/train_bdd.py --num-gpus 3 --dist-url auto --eval-only MODEL.WEIGHTS ./out/CondInst_r_101_3x_multi_scale_6bs_gradient_clip/model_0116499.pth