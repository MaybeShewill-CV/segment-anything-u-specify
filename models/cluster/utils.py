#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-10 下午3:25
# @Author  : MaybeShewill-CV
# @Site    :  
# @File    : utils.py
# @IDE: PyCharm Community Edition
"""
utils func
"""
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_color_pool(color_nums):
    """
    随机生成颜色池, 用来给不同的车道线染不同的颜色
    :param color_nums: 需要生成的颜色池大小
    :return:
    """
    color_pool = []

    np.random.seed(int(time.time()))

    for i in range(color_nums):
        b = int(np.random.randint(0, 255, dtype=np.uint8))
        g = int(np.random.randint(0, 255, dtype=np.uint8))
        r = int(np.random.randint(0, 255, dtype=np.uint8))
        color_pool.append((b, g, r))

    return color_pool


def generate_sam_mask_images(sam_masks, bbox_cls_ids=None, draw_bbox=False):
    """

    :param sam_masks:
    :param bbox_cls_ids:
    :param draw_bbox:
    :return:
    """
    seg_images = sam_masks['segmentations']
    bboxes = sam_masks['bboxes']
    cls_ids = bbox_cls_ids if bbox_cls_ids is not None else sam_masks['bbox_ori_ids']
    color_pool = generate_color_pool(color_nums=max(cls_ids) + 1)

    mask_image = np.ones(shape=(seg_images[0].shape[0], seg_images[0].shape[1], 3), dtype=np.uint8)
    for idx, cls_id in enumerate(cls_ids):
        color = color_pool[cls_id]
        # draw bbox
        if draw_bbox:
            bbox_pt1 = [bboxes[idx][0], bboxes[idx][1]]
            bbox_pt1 = [int(tmp) for tmp in bbox_pt1]
            bbox_pt2 = [bboxes[idx][0] + bboxes[idx][2], bboxes[idx][1] + bboxes[idx][3]]
            bbox_pt2 = [int(tmp) for tmp in bbox_pt2]
            cv2.rectangle(mask_image, bbox_pt1, bbox_pt2, color, 2)
        # draw mask
        mask_image[:, :, 0][seg_images[idx]] = color[0]
        mask_image[:, :, 1][seg_images[idx]] = color[1]
        mask_image[:, :, 2][seg_images[idx]] = color[2]

    return mask_image


def show_anns(anns, bbox_cls_ids=None):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    cls_ids = bbox_cls_ids if bbox_cls_ids is not None else anns['bbox_ori_ids']
    color_pool = generate_color_pool(color_nums=max(cls_ids) + 1)
    for idx, m in enumerate(anns['segmentations']):
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = color_pool[cls_ids[idx]]
        for i in range(3):
            img[:, :, i] = color_mask[i] / 255
        ax.imshow(np.dstack((img, m*0.35)))

    return
