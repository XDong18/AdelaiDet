export CUDA_VISIBLE_DEVICES=6,7,8,9
python train_bdd.py --num-gpus 4 --dist-url auto --eval-only MODEL.WEIGHTS ./bdd100k_retinanet_new/Retinanet_r_101_fpn_1x-745c57cc.pth