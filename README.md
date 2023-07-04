# Segment-Anything-U-Specify
Use SAM and CLIP model to segment unique instances you want.
You may use this repo to segment any instances in the picture with
text prompts.

The main network architecture is as follows:

`Clip Model Architecture`
![CLIP_MODEL](./data/resources/clip_model.png)

`SAM Model Architecture`
![SAM](./data/resources/sam_model.png)

## Installation

Install python packages via commands:
```
pip3 install -r requirements.txt
```
Download pretrained model weights
```
cd PROJECT_ROOT_DIR
bash scripts/download_pretrained_ckpt.sh
```

## Instance Segmentation With Text Prompts
Instance segmentor first using sam model to get all obj's mask of the input image. Second using clip model to classify each mask with both
image features and your text prompts features.

```
cd PROJECT_ROOT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/sam_clip_text_seg.py --input_image_path ./data/test_images/test_bear.jpg --text bear
```

`Bear Instance Segmentation Result, Text Prompt: bear`
![bear_insseg_result](./data/resources/test_bear_insseg_result.jpg)

`Athelete Instance Segmentation Result, Text Prompt: athlete`
![athlete_insseg_result](./data/resources/test_baseball_insseg_result.jpg)

`Horse Instance Segmentation Result, Text Prompt: horse`
![horse_insseg_result](./data/resources/test_horse_insseg_result_after.jpg)

`Dog Instance Segmentation Result, Text Prompt: dog`
![dog_insseg_result](./data/resources/test_dog_insseg_result.jpg)

`Fish Instance Segmentation Result, Text Prompt: fish`
![fish_insseg_result](./data/resources/test_fish_insseg_result.jpg)

`Strawberry Instance Segmentaton Result, Text Prompt: strawberry`
![strawberry_insseg_result](./data/resources/test_strawberry_insseg_result.jpg)

`Glasses Instance Segmentaton Result, Text Prompt: glasses`
![glasses_insseg_result](./data/resources/test_glasses_insseg_result.jpg)

`Tv Instance Segmentaton Result, Text Prompt: television`
![tv_insseg_result](./data/resources/test_tv_insseg_result.jpg)

`Shoes Instance Segmentaton Result, Text Prompt: shoe`
![shoes_insseg_result](./data/resources/test_shoes_insseg_result.jpg)

`Bridge Instance Segmentaton Result, Text Prompt: bridge`
![bridge_insseg_result](./data/resources/test_bridge_insseg_result.jpg)

`Airplane Instance Segmentaton Result, Text Prompt: airplane`
![airplane_insseg_result](./data/resources/test_airplane_insseg_result.jpg)

### Support Multiple Classes Segmentation All In Once ---- YOSO ---- You Only Segment Once
```
cd PROJECT_ROOT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/sam_clip_text_seg.py --input_image_path ./data/test_images/test_horse.jpg --text "horse,mountain,grass,sky,clouds,tree" --cls_score_thresh 0.5 --use_text_prefix
```

`Horse Instance Segmentation Result, Text Prompt: horse,mountain,grass,sky,clouds,tree`
![horse_insseg_result](./data/resources/test_horse_insseg_result_muti_label.jpg)
`Tv Instance Segmentaton Result, Text Prompt: television,audio system,tape recorder,box`
![tv_insseg_result](./data/resources/test_tv_insseg_result_multi_label.jpg)
`Strawberry Instance Segmentaton Result, Text Prompt: strawberry,grapefruit,spoon,wolfberry,oatmeal`
![strawberry_insseg_result](./data/resources/test_strawberry_insseg_result_multi_label.jpg)
`Frog Instance Segmentaton Result, Text Prompt: frog,turtle,snail,eye`
![frog_insseg_result](./data/resources/test_frog_insseg_result_multi_label.jpg)

#### Instance Segmentation Provement

##### 2023-04-21 improve background segmentation problem

`Befor Optimize`
![before](./data/resources/test_horse_insseg_result.jpg)
`After Optimize`
![after](./data/resources/test_horse_insseg_result_after.jpg)

## Unsupervised Cluster Semantic Objects From SAM Model
Cluster first using sam model to get all obj's mask of the input image. Second using clip model to extract image features for each objects. Third calculate feature distance of every two object pairs. Finally using a similarity threshold to cluster source objects.

To test the cluster simply run

```
cd PROJECT_ROOT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH
python tools/cluster_sam.py --input_image_path ./data/test_images/test_bear.jpg --simi_thresh 0.82
```

`Bear Cluster Result`
![bear_cluster_result](./data/resources/test_bear_result.jpg)

`Horse Cluster Result`
![horse_cluster_result](./data/resources/test_horse_result.jpg)

Each row represents `source image`, `sam origin mask`, `ori masked image`, `clustered mask`, `cluster masked image`

## UPDATES

### 2023-07-04 Integrate MobileSAM

Integrate MobileSAM into the pipeline for lightweight and faster inference. If you want to use mobile-sam to segment your
image all you need to do is to modify `./config/sam.yaml` file. Modify the model name field to `vit_t` and modify the 
model weight file path to `./pretrained/sam/mobile_sam.pt`

## TODO
- [x] Test different kinds of cluster method
- [x] Using cluster result as input prompts to reseg the image via sam model
- [ ] Merge embedding feats of global image and masked image

## Acknowledgement

Most of the repo's code borrows from opeai's clip repo and facebook's segment-anything repo:

- [CLIP](https://github.com/openai/CLIP)
- [segment-anything](https://github.com/facebookresearch/segment-anything)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MaybeShewill-CV/segment-anything-u-specify&type=Date)](https://star-history.com/#MaybeShewill-CV/segment-anything-u-specify&Date)

## Visitor Count

![Visitor Count](https://profile-counter.glitch.me/15725187_sam_clip/count.svg)

## Contact

