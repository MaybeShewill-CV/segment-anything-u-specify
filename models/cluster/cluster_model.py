#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-10 下午1:57
# @Author  : MaybeShewill-CV
# @Site    :  
# @File    : cluster_model.py
# @IDE: PyCharm Community Edition
"""
cluster model
"""
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from models.cluster import utils
from models.clip import tokenize
from models import build_clip_model
from models import build_sam_mask_generator


class SamClipCluster(object):
    """
    cluster segment objects from sam's output
    """
    def __init__(self, sam_cfg, clip_cfg, cluster_cfg):
        """

        :param sam_cfg:
        :param clip_cfg:
        :param cluster_cfg:
        """
        self.mask_generator = build_sam_mask_generator(sam_cfg)
        self.clip_model, self.clip_preprocess = build_clip_model(clip_cfg)
        self.device = torch.device(cluster_cfg.MODEL.DEVICE)
        self.top_k_objs = cluster_cfg.CLUSTER.TOP_K_MASK_COUNT
        self.similarity_thresh = cluster_cfg.CLUSTER.SIMILARITY_THRESH
        self.max_input_size = cluster_cfg.CLUSTER.MAX_INPUT_SIZE
        self.imagenet_cls_text_prompts = utils.generate_imagenet_classification_text_prompts()
        self.imagenet_cls_text_token = tokenize(self.imagenet_cls_text_prompts).to(self.device)

    def _generate_sam_mask(self, input_image: np.ndarray):
        """

        :param input_image:
        :return:
        """
        masks = self.mask_generator.generate(input_image)
        most_stable_mask = sorted(masks, key=lambda d: d['area'])
        if len(most_stable_mask) > self.top_k_objs:
            most_stable_mask = most_stable_mask[-self.top_k_objs:]
        sam_masks = {
            'segmentations': [tmp['segmentation'] for tmp in most_stable_mask],
            'bboxes': [tmp['bbox'] for tmp in most_stable_mask],
            'stability_scores': [tmp['stability_score'] for tmp in most_stable_mask],
        }
        return sam_masks

    def _extract_image_features(self, input_image: np.ndarray, normalize=False):
        """

        :param input_image:
        :return:
        """
        image = Image.fromarray(input_image)
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.clip_model.encode_image(image)
        image_features = F.normalize(image_features, dim=-1) if normalize else image_features
        image_features = image_features.squeeze(0)

        return image_features.cpu().numpy()

    def _classify_image(self, input_image: np.ndarray):
        """

        :param input_image:
        :return:
        """
        image = Image.fromarray(input_image)
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        logits_per_image, logits_per_text = self.clip_model(image, self.imagenet_cls_text_token)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0, :]
        cls_id = np.argmax(probs)
        cls_name = self.imagenet_cls_text_prompts[cls_id].replace('a photo of a', '')

        return cls_name

    def _extract_mask_features(self, input_image, mask):
        """

        :param input_image:
        :param mask:
        :return:
        """
        bboxes_features = []
        bboxes_ids = []
        for idx, bbox in enumerate(mask['bboxes']):
            bbox = [int(tmp) for tmp in bbox]
            roi_image = input_image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
            roi_features = self._extract_image_features(roi_image, False)
            bboxes_features.append(roi_features)
            bboxes_ids.append(idx)

        mask['bbox_features'] = np.asarray(bboxes_features, dtype=np.float32)
        mask['bbox_ori_ids'] = bboxes_ids
        return

    def _classify_mask(self, input_image, mask):
        """

        :param input_image:
        :param mask:
        :return:
        """
        bboxes_cls_names = []
        for idx, bbox in enumerate(mask['bboxes']):
            bbox = [int(tmp) for tmp in bbox]
            roi_image = input_image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2], :]
            cls_name = self._classify_image(roi_image)
            bboxes_cls_names.append(cls_name)

        mask['bbox_cls_names'] = bboxes_cls_names
        return

    def _cluster_bbox_features(self, bbox_features: np.ndarray):
        """

        :param bbox_features: [N, 512] feats
        :return:
        """
        norm = np.linalg.norm(bbox_features, axis=1)
        similarity_matrix = np.dot(bbox_features, bbox_features.T) / np.outer(norm, norm)
        similar_obj = np.argwhere(similarity_matrix > self.similarity_thresh)

        obj_classes = [i for i in range(bbox_features.shape[0])]
        for i, j in similar_obj:
            if i != j and i in obj_classes and j in obj_classes:
                obj_classes[j] = obj_classes[i]

        return obj_classes

    def cluster_image(self, input_image_path):
        """

        :param input_image_path:
        :return:
        """
        # read input image
        input_image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
        if input_image.shape[0] > self.max_input_size[0] or input_image.shape[1] > self.max_input_size[1]:
            h, w, _ = input_image.shape
            hw_ratio = h / w if h > w else w / h
            if h > w:
                dsize = (int(self.max_input_size[1] / hw_ratio), self.max_input_size[1])
            else:
                dsize = (self.max_input_size[0], int(self.max_input_size[0] / hw_ratio))
            input_image = cv2.resize(input_image, dsize=dsize)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # extract mask from sam model
            masks = self._generate_sam_mask(input_image)
            # extract each mask's features
            self._extract_mask_features(input_image, masks)
            # classify each mask's label
            self._classify_mask(input_image, masks)

        # cluster obj ids
        cluster_obj_ids = self._cluster_bbox_features(bbox_features=masks['bbox_features'])
        masks['bbox_cluster_ids'] = cluster_obj_ids

        # diff mask image
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        ori_mask = utils.generate_sam_mask_images(
            masks,
            bbox_cls_ids=masks['bbox_ori_ids'],
            bbox_cls_names=masks['bbox_cls_names'],
            draw_bbox=True
        )
        ori_mask_add = cv2.addWeighted(input_image, 0.5, ori_mask, 0.5, 0.0)
        clustered_mask = utils.generate_sam_mask_images(masks, bbox_cls_ids=masks['bbox_cluster_ids'])
        clustered_mask_add = cv2.addWeighted(input_image, 0.5, clustered_mask, 0.5, 0.0)

        ret = {
            'source': input_image,
            'ori_mask': ori_mask,
            'ori_mask_add': ori_mask_add,
            'cluster_mask': clustered_mask,
            'cluster_mask_add': clustered_mask_add
        }

        return ret
