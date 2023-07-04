#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-10 上午11:50
# @Author  : MaybeShewill-CV
# @Site    :  
# @File    : __init__.py.py
# @IDE: PyCharm Community Edition
"""
model
"""
import torch

from local_utils.config_utils import parse_config_utils
from models.clip import clip
import models.sam as sam_builder


def build_sam_model(cfg):
    """

    :param cfg:
    :return:
    """
    supported_model_name = ['vit_h', 'vit_l', 'vit_b', 'vit_t', 'default']
    model_name = cfg.MODEL.MODEL_NAME
    if model_name not in supported_model_name:
        raise ValueError('not supported model: {:s}, only supported {}'.format(model_name, supported_model_name))
    ckpt_path = cfg.MODEL.CKPT_PATH
    device = torch.device(cfg.MODEL.DEVICE)
    build_func = sam_builder.sam_model_registry[model_name]
    model = build_func(checkpoint=ckpt_path)
    return model.to(device)


def build_sam_mask_generator(cfg):
    """

    :param cfg:
    :return:
    """
    sam = build_sam_model(cfg)
    points_per_side = cfg.MASK_GENERATOR.PTS_PER_SIDE
    points_per_batch = cfg.MASK_GENERATOR.PTS_PER_BATCH
    pred_iou_thresh = cfg.MASK_GENERATOR.PRED_IOU_THRESH
    stability_score_thresh = cfg.MASK_GENERATOR.STABILITY_SCORE_THRESH
    stability_score_offset = cfg.MASK_GENERATOR.STABILITY_SCORE_THRESH
    box_nms_thresh = cfg.MASK_GENERATOR.BOX_NMS_THRESH
    crop_n_layers = cfg.MASK_GENERATOR.CROP_N_LAYERS
    crop_nms_thresh = cfg.MASK_GENERATOR.CROP_NMS_THRESH
    crop_overlap_ratio = cfg.MASK_GENERATOR.CROP_OVERLAP_RATIO
    crop_n_points_downscale_factor = cfg.MASK_GENERATOR.CROP_N_POINTS_DOWNSCALE_FACTOR
    point_grids = None if cfg.MASK_GENERATOR.POINT_GRIDS.lower() == 'none' else cfg.MASK_GENERATOR.POINT_GRIDS
    min_mask_region_area = cfg.MASK_GENERATOR.MIN_MASK_REGION_AERA
    output_mode = cfg.MASK_GENERATOR.OUTPUT_MODE
    mask_generator = sam_builder.SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        stability_score_offset=stability_score_offset,
        box_nms_thresh=box_nms_thresh,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        point_grids=point_grids,
        min_mask_region_area=min_mask_region_area,
        output_mode=output_mode
    )
    return mask_generator


def build_clip_model(cfg):
    """

    :param cfg:
    :return:
    """
    ckpt_dir = cfg.MODEL.CKPT_DIR
    device = cfg.MODEL.DEVICE
    model_name = cfg.MODEL.NAME
    model, preprocess = clip.load(model_name, device=device, download_root=ckpt_dir)

    return model, preprocess


import models.cluster as sam_clip_cluster


def build_cluster(cfg):
    """

    :param cfg:
    :return:
    """
    sam_cfg_path = cfg.MODEL.SAM.CFG_PATH
    clip_cfg_path = cfg.MODEL.CLIP.CFG_PATH
    sam_cfg = parse_config_utils.Config(config_path=sam_cfg_path)
    clip_cfg = parse_config_utils.Config(config_path=clip_cfg_path)
    model = sam_clip_cluster.cluster_model.SamClipCluster(
        sam_cfg=sam_cfg,
        clip_cfg=clip_cfg,
        cluster_cfg=cfg
    )

    return model


import models.detector as sam_clip_insseg


def build_sam_clip_text_ins_segmentor(cfg):
    """

    :param cfg:
    :return:
    """
    sam_cfg_path = cfg.MODEL.SAM.CFG_PATH
    clip_cfg_path = cfg.MODEL.CLIP.CFG_PATH
    sam_cfg = parse_config_utils.Config(config_path=sam_cfg_path)
    clip_cfg = parse_config_utils.Config(config_path=clip_cfg_path)
    model = sam_clip_insseg.insseg_model.SamClipInsSegmentor(
        sam_cfg=sam_cfg,
        clip_cfg=clip_cfg,
        insseg_cfg=cfg
    )

    return model
