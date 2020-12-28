export CUDA_VISIBLE_DEVICES=4,5,8
python vis/demo.py --config-file configs/BDD00K-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
  --input /data5/bdd100k/images/10k/val/* \
  --output show/mask_rcnn_R_50_FPN_3x_val/ \
  --opts SOLVER.IMS_PER_BATCH 12 MODEL.WEIGHTS /shared/xudongliu/code/bdd100k_det/bdd100k_maskrcnn_720_8bs_0.01_3x/model_final.pth