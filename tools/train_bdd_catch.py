import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
# from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator


logger = logging.getLogger("adet")

class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def resume_or_load(self, resume=True):
        if not isinstance(self.checkpointer, AdetCheckpointer):
            # support loading a few other backbones
            self.checkpointer = AdetCheckpointer(
                self.model,
                self.cfg.OUTPUT_DIR,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        super().resume_or_load(resume=resume)

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_val")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # set config file
    cfg.merge_from_file("configs/CondInst/MS_R_101_3x_sem_bdd100k_single_scale.yaml")
    cfg.DATASETS.TRAIN = ("bdd100k_train",)
    cfg.DATASETS.TEST = ("bdd100k_test",)
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo


    cfg.OUTPUT_DIR = './out/CondInst_R_101_3x_sem_bdd100k_single_scale_catch_3'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # new added solver arguments
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.TEST.EVAL_PERIOD = 500
    # end of new arguments

    # cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")
    return cfg

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = AdetCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    start_iter = (
        0
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    mapper = DatasetMapperWithBasis(cfg, True)
    data_loader = build_detection_train_loader(cfg, mapper=mapper)
    # data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            # catch loss infinite
            if not torch.isfinite(losses).all():
                print(f'!!!infinite loss batch catched in rank:{comm.get_rank()}!!!')
                if comm.is_main_process():
                    pass
                    # torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'catched_cpt.pth'))
                
                # torch.save(data, os.path.join(cfg.OUTPUT_DIR, f'catched_data_{comm.get_rank()}.pth'))
                # torch.save(loss_dict, os.path.join(cfg.OUTPUT_DIR, f'catched_loss_{comm.get_rank()}.pth'))
                exit(0)
            
            assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                

            optimizer.zero_grad()
            losses.backward()
            
            # compute max abs(gradient)
            if comm.is_main_process():
                temp_max = 0
                for para in model.parameters():
                    if (para.grad!=None).all():
                        temp_max = max(temp_max, torch.max(torch.abs(para.grad)).item())
                storage.put_scalar("max_gradient", temp_max, smoothing_hint=False)

            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # if (
            #     cfg.TEST.EVAL_PERIOD > 0
            #     and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
            #     and iteration != max_iter - 1
            # ):
                # do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                # comm.synchronize()

            # if iteration - start_iter > 5 and (
            #     (iteration + 1) % 1 == 0 or iteration == max_iter - 1
            # ):
            for writer in writers:
                writer.write()
            periodic_checkpointer.step(iteration)

def main(args):
    cfg = setup(args)
    # dataset
    register_coco_instances("bdd100k_train", {}, "/shared/xudongliu/bdd100k/labels/ins_seg/ins_seg_train.json", "/shared/xudongliu/bdd100k/10k/train")
    register_coco_instances("bdd100k_test", {}, "/shared/xudongliu/bdd100k/labels/ins_seg/ins_seg_val.json", "/shared/xudongliu/bdd100k/10k/val")

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        # return res
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)


if __name__ == "__main__":
    # dataset registration
    # global DatasetCatalog, MetadataCatalog


    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )