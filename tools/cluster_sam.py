#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-10 下午7:33
# @Author  : MaybeShewill-CV
# @Site    :  
# @File    : cluster_sam.py
# @IDE: PyCharm Community Edition
"""
cluster sam segmentation result with clip features
"""
import os
import os.path as ops
import argparse

import cv2

from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils
from models import build_cluster


LOG = init_logger.get_logger('cluster.log')


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='./data/test_bear.jpg', required=True)
    parser.add_argument('--cluster_cfg_path', type=str, default='./config/cluster.yaml')
    parser.add_argument('--save_dir', type=str, default='./output/cluster')
    parser.add_argument('--simi_thresh', type=float, default=None)

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
    cluster_cfg_path = args.cluster_cfg_path
    if not ops.exists(cluster_cfg_path):
        LOG.error('input cluster cfg path: {:s} not exists'.format(cluster_cfg_path))
        return
    cluster_cfg = parse_config_utils.Config(config_path=cluster_cfg_path)
    if args.simi_thresh is not None:
        cluster_cfg.CLUSTER.SIMILARITY_THRESH = float(args.simi_thresh)

    # init cluster
    LOG.info('Start initializing cluster ...')
    cluster = build_cluster(cfg=cluster_cfg)
    LOG.info('Cluster initialized complete')
    LOG.info('Start to segment and cluster input image ...')
    ret = cluster.cluster_image(input_image_path)
    LOG.info('segment and cluster complete')

    # save cluster result
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    ori_image_save_path = ops.join(save_dir, input_image_name)
    cv2.imwrite(ori_image_save_path, ret['source'])
    ori_mask_save_path = ops.join(save_dir, '{:s}_ori_mask.png'.format(input_image_name.split('.')[0]))
    cv2.imwrite(ori_mask_save_path, ret['ori_mask'])
    ori_mask_add_save_path = ops.join(save_dir, '{:s}_ori_mask_add.png'.format(input_image_name.split('.')[0]))
    cv2.imwrite(ori_mask_add_save_path, ret['ori_mask_add'])
    clustered_mask_save_path = ops.join(save_dir, '{:s}_clustered_mask.png'.format(input_image_name.split('.')[0]))
    cv2.imwrite(clustered_mask_save_path, ret['cluster_mask'])
    clustered_mask_add_save_path = ops.join(
        save_dir,
        '{:s}_clustered_mask_add.png'.format(input_image_name.split('.')[0])
    )
    cv2.imwrite(clustered_mask_add_save_path, ret['cluster_mask_add'])
    LOG.info('save segment and cluster result into {:s}'.format(save_dir))

    return


if __name__ == '__main__':
    """
    main func
    """
    main()
