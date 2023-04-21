#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-11 下午2:15
# @Author  : MaybeShewill-CV
# @Site    :  
# @File    : sam_clip_text_seg.py
# @IDE: PyCharm Community Edition
"""
instance segmentation image with sam and clip with text prompts
"""
import os
import os.path as ops
import argparse

import cv2

from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils
from models import build_sam_clip_text_ins_segmentor


LOG = init_logger.get_logger('instance_seg.log')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='./data/test_bear.jpg', required=True)
    parser.add_argument('--insseg_cfg_path', type=str, default='./config/insseg.yaml')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--cls_score_thresh', type=float, default=None)
    parser.add_argument('--save_dir', type=str, default='./output/insseg')
    parser.add_argument('--use_text_prefix', action='store_true')

    return parser.parse_args()


def main():
    """

    :return:
    """
    # init args
    args = init_args()
    input_image_path = args.input_image_path
    input_image_name = ops.split(input_image_path)[1]
    if not ops.exists(input_image_path):
        LOG.error('input image path: {:s} not exists'.format(input_image_path))
        return
    insseg_cfg_path = args.insseg_cfg_path
    if not ops.exists(insseg_cfg_path):
        LOG.error('input innseg cfg path: {:s} not exists'.format(insseg_cfg_path))
        return
    insseg_cfg = parse_config_utils.Config(config_path=insseg_cfg_path)
    if args.text is not None:
        unique_labels = args.text.split(',')
    else:
        unique_labels = None
    if args.cls_score_thresh is not None:
        insseg_cfg.INS_SEG.CLS_SCORE_THRESH = args.cls_score_thresh
    use_text_prefix = True if args.use_text_prefix else False

    # init cluster
    LOG.info('Start initializing instance segmentor ...')
    segmentor = build_sam_clip_text_ins_segmentor(cfg=insseg_cfg)
    LOG.info('Segmentor initialized complete')
    LOG.info('Start to segment input image ...')
    ret = segmentor.seg_image(input_image_path, unique_label=unique_labels, use_text_prefix=use_text_prefix)
    LOG.info('segment complete')

    # save cluster result
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    ori_image_save_path = ops.join(save_dir, input_image_name)
    cv2.imwrite(ori_image_save_path, ret['source'])
    mask_save_path = ops.join(save_dir, '{:s}_insseg_mask.png'.format(input_image_name.split('.')[0]))
    cv2.imwrite(mask_save_path, ret['ins_seg_mask'])
    mask_add_save_path = ops.join(save_dir, '{:s}_insseg_add.png'.format(input_image_name.split('.')[0]))
    cv2.imwrite(mask_add_save_path, ret['ins_seg_add'])

    LOG.info('save segment and cluster result into {:s}'.format(save_dir))

    return


if __name__ == '__main__':
    """
    main func
    """
    main()
