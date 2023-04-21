#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023-04-11 下午1:49
# @Author  : MaybeShewill-CV
# @Site    :  
# @File    : insseg_model.py
# @IDE: PyCharm Community Edition
"""
instance segmentation model with sam and clip
"""
import numpy as np
import cv2
from PIL import Image
import torch

from models.detector import utils
from models.clip import tokenize
from models import build_clip_model
from models import build_sam_mask_generator


class SamClipInsSegmentor(object):
    """

    """
    def __init__(self, sam_cfg, clip_cfg, insseg_cfg):
        """

        :param sam_cfg:
        :param clip_cfg:
        :param insseg_cfg:
        """
        self.mask_generator = build_sam_mask_generator(sam_cfg)
        self.clip_model, self.clip_preprocess = build_clip_model(clip_cfg)
        self.device = torch.device(insseg_cfg.MODEL.DEVICE)
        self.top_k_objs = insseg_cfg.INS_SEG.TOP_K_MASK_COUNT
        self.cls_score_thresh = insseg_cfg.INS_SEG.CLS_SCORE_THRESH
        self.max_input_size = insseg_cfg.INS_SEG.MAX_INPUT_SIZE
        self.obj365_text_prompts = utils.generate_object365_text_prompts()
        self.obj365_text_token = tokenize(self.obj365_text_prompts).to(self.device)
        self.text_token = None

    def _set_text_tokens(self, texts):
        """

        :param texts:
        :return:
        """
        self.text_token = tokenize(texts=texts).to(self.device)
        return

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

    @staticmethod
    def _crop_rotate_image_roi(input_image, seg_mask):
        """

        :param input_image:
        :param seg_mask:
        :return:
        """
        y, x = np.where(seg_mask == 1)
        fg_pts = np.vstack((x, y)).transpose()
        src_image = cv2.bitwise_or(input_image, input_image, mask=np.asarray(seg_mask, dtype=np.uint8))
        roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(fg_pts)
        extend_size = 20
        if roi_x - extend_size >= 0:
            roi_x -= extend_size
            roi_w += extend_size
        if roi_y - extend_size >= 0:
            roi_y -= extend_size
            roi_h += extend_size
        roi_image = src_image[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w, :]
        if np.any(np.shape(roi_image) < (3, 3)):
            return None
        return roi_image

    def _classify_image(self, input_image: np.ndarray, text=None):
        """

        :param input_image:
        :return:
        """
        image = Image.fromarray(input_image)
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        if text is None:
            logits_per_image, logits_per_text = self.clip_model(image, self.obj365_text_token)
        else:
            if self.text_token is None:
                text_token = tokenize(texts=text).to(self.device)
                logits_per_image, logits_per_text = self.clip_model(image, text_token)
            else:
                logits_per_image, logits_per_text = self.clip_model(image, self.text_token)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0, :]
        cls_id = np.argmax(probs)
        score = probs[cls_id]
        if text is None:
            if score < 0.15:
                cls_id = probs.shape[0] - 1
            return cls_id
        else:
            if score < self.cls_score_thresh:
                cls_id = probs.shape[0] - 1
            return cls_id

    def _classify_mask(self, input_image, mask, text=None):
        """

        :param input_image:
        :param mask:
        :return:
        """
        bboxes_cls_names = []
        for idx, bbox in enumerate(mask['bboxes']):
            roi_image = self._crop_rotate_image_roi(input_image, mask['segmentations'][idx])
            if roi_image is None:
                cls_name = 'background'
                bboxes_cls_names.append(cls_name)
                continue
            # cv2.imwrite('{:d}_mask.png'.format(idx), roi_image[:, :, (2, 1, 0)])
            cls_id = self._classify_image(roi_image, text=text)
            if text is None:
                cls_name = self.obj365_text_prompts[cls_id].split('a photo of')[1].strip(' ')
                bboxes_cls_names.append(cls_name)
            else:
                cls_name = text[cls_id]
                if cls_name.startswith('a photo of'):
                    cls_name = cls_name.split(' ')[3]
                bboxes_cls_names.append(cls_name)

        mask['bbox_cls_names'] = bboxes_cls_names
        return

    def seg_image(self, input_image_path, unique_label=None, use_text_prefix=False):
        """

        :param input_image_path:
        :param unique_label:
        :param use_text_prefix:
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
            # classify each mask's label
            if unique_label is None:
                self._classify_mask(input_image, masks, text=None)
            else:
                texts = utils.generate_text_prompts_for_instance_seg(
                    unique_labels=unique_label,
                    use_text_prefix=use_text_prefix
                )
                self._set_text_tokens(texts)
                self._classify_mask(input_image, masks, text=texts)

        # visualize segmentation result
        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        ins_seg_mask = utils.visualize_instance_seg_results(masks, draw_bbox=True)
        ins_seg_add = cv2.addWeighted(input_image, 0.5, ins_seg_mask, 0.5, 0.0)

        ret = {
            'source': input_image,
            'ins_seg_mask': ins_seg_mask,
            'ins_seg_add': ins_seg_add
        }

        return ret
