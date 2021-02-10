export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
python tools/train_bdd_catch.py --num-gpus 8 --dist-url tcp://0.0.0.0:12340 \
--eval-only MODEL.WEIGHTS ./out/CondInst_R_101_3x_sem_bdd100k_single_scale_catch_3/model_final.pth