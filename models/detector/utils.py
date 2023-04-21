#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-11 下午1:55
# @Author  : MaybeShewill-CV
# @Site    :  
# @File    : utils.py
# @IDE: PyCharm Community Edition
"""

"""
import time

import cv2
import numpy as np


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


def visualize_instance_seg_results(sam_masks, draw_bbox=False):
    """

    :param sam_masks:
    :param draw_bbox:
    :return:
    """
    seg_images = sam_masks['segmentations']
    bboxes = sam_masks['bboxes']
    bboxes_names = sam_masks['bbox_cls_names']
    unique_names = list(np.unique(bboxes_names))
    color_pool = generate_color_pool(color_nums=len(unique_names) + 1)

    mask_image = np.ones(shape=(seg_images[0].shape[0], seg_images[0].shape[1], 3), dtype=np.uint8)
    for idx, _ in enumerate(bboxes):
        if bboxes_names[idx] == 'background':
            continue
        color_id = unique_names.index(bboxes_names[idx])
        color = color_pool[color_id]
        # draw mask
        mask_image[:, :, 0][seg_images[idx]] = color[0]
        mask_image[:, :, 1][seg_images[idx]] = color[1]
        mask_image[:, :, 2][seg_images[idx]] = color[2]
    # draw bbox
    if draw_bbox:
        for idx, _ in enumerate(bboxes):
            if bboxes_names[idx] == 'background':
                continue
            color_id = unique_names.index(bboxes_names[idx])
            color = color_pool[color_id]
            bbox_pt1 = [bboxes[idx][0], bboxes[idx][1]]
            bbox_pt1 = [int(tmp) for tmp in bbox_pt1]
            bbox_pt2 = [bboxes[idx][0] + bboxes[idx][2], bboxes[idx][1] + bboxes[idx][3]]
            bbox_pt2 = [int(tmp) for tmp in bbox_pt2]
            cv2.rectangle(mask_image, bbox_pt1, bbox_pt2, color, 2)
            text = bboxes_names[idx].split(',')[0]
            org = [bbox_pt1[0] - 10, bbox_pt1[1] - 10]
            cv2.putText(mask_image, text, org, cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

    return mask_image


def generate_imagenet_classification_text_prompts():
    """

    :return:
    """
    text_prefix = 'a photo of'
    text_prompts = open('./data/resources/imagenet-classes.txt', 'r').readlines()
    text_prompts = list(map(lambda x: x.rstrip('\r').rstrip('\n'), text_prompts))
    text_prompts = list(map(lambda x: ' '.join([text_prefix, x]), text_prompts))

    return text_prompts


def generate_object365_text_prompts():
    """

    :return:
    """
    text_prefix = 'a photo of'
    text_prompts = open('./data/resources/obj365-classes.txt', 'r').readlines()
    text_prompts = list(map(lambda x: x.rstrip('\r').rstrip('\n').split(' ')[0], text_prompts))
    text_prompts = list(map(lambda x: ' '.join([text_prefix, x]), text_prompts))

    return text_prompts


def generate_text_prompts_for_instance_seg(unique_labels, use_text_prefix=True):
    """

    :param unique_labels:
    :param use_text_prefix:
    :return:
    """
    if unique_labels[-1] != 'background':
        unique_labels.append('background')
    text_pre = 'a photo of {:s}'
    if use_text_prefix:
        text_prompts = [text_pre.format(tmp) for tmp in unique_labels]
    else:
        text_prompts = list(unique_labels)
    return text_prompts
