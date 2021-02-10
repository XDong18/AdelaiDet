export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9
python vis/demo.py --config-file configs/CondInst/MS_R_101_3x_sem_bdd100k_single_scale.yaml \
  --input /shared/xudongliu/bdd100k/10k/val/* \
  --output ./show/condInst_r101_single_scale_3x_val/ \
  --opts SOLVER.IMS_PER_BATCH 16 MODEL.WEIGHTS ./out/CondInst_R_101_3x_sem_bdd100k_single_scale_catch_3/model_final.pth