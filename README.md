This is the code base for CVPR2022 paper "**Learning to prompt for Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model**
# Introduction
Recently, vision-language pre-training shows great potential in open-vocabulary object detection, where detectors trained on base classes are devised for detecting new classes. The class text embedding is firstly generated by feeding class descriptions of interest, known as prompt, to the text encoder of a pre-trained vision-language model. It is then used as the region classifier to supervise the training of a detector. The key element that leads to the success of this model is the proper prompt, which requires careful words tuning and ingenious design. To avoid laborious prompt engineering, there are some prompt representation learning methods being proposed for the image classification task, which however can only be sub-optimal solutions when applied to the detection task. In this paper,  we introduce a novel method, detection prompt (DetPro), to learn continuous prompt representations for open-vocabulary object detection based on the pre-trained vision-language model. Different from the previous classification-oriented method, DetPro has two highlights in its design: 1) a background interpretation scheme to include the proposals in image background into the prompt training; 2) a context grading scheme to separate proposals in image foreground for tailored prompt training. We assemble DetPro with ViLD, a recent state-of-the-art open-world object detector, and conduct experiments on the LVIS as well as transfer learning on the Pascal VOC, COCO, Objects365 datasets. Experimental results show that our DetPro outperforms the baseline ViLD in all settings, e.g., +3.4 AP$^{\text{box}}$ and +3.0 AP$^{\text{mask}}$ improvements on the novel classes of LVIS.
# Prepare data
Download dataset according to [LVIS](https://www.lvisdataset.org/) [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) [COCO](https://cocodataset.org/#home) [Objects365](https://www.objects365.org/overview.html). It is recommended to download and extract the dataset somewhere outside the project directory and symlink the dataset root to data as below
```
├── mmdet
├── tools
├── configs
├── data
├── ├── lvis_v1
├── ├── ├──annotations
├── ├── ├──train2017
├── ├── ├──val2017
├── ├── ├──proposals
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
│   ├── objects365
│   │   ├── annotations
│   │   ├── train
│   │   ├── val



```
# Main Results
| Model | Lr Schd | Box AP &nbsp;&nbsp;&nbsp;| Mask AP &nbsp;&nbsp;&nbsp; | Config | Prompt | Model |
| ---- | ---- | ---- | ------- | ---- | ---- | ---- |
| Vild * | 20e | 17.4 27.5 31.9 27.5 &nbsp;&nbsp;&nbsp;| 16.8 25.6 28.5 25.2 &nbsp;&nbsp;&nbsp;|[config](https://github.com/dyabel/detpro/blob/main/configs/lvis/detpro_ens_2x.py) | [prompt](https://cloud.tsinghua.edu.cn/f/3f9017c3e217496ebc25/?dl=1)|[model](https://cloud.tsinghua.edu.cn/f/d57e11e2ebf24d509218/?dl=1)  |
| detpro | 20e | 20.8 27.8 32.4 28.4 &nbsp;&nbsp;&nbsp;| 19.8 25.6 28.9 25.9 &nbsp;&nbsp;&nbsp;|[config](https://github.com/dyabel/detpro/blob/main/configs/lvis/detpro_ens_2x.py) | [prompt](https://cloud.tsinghua.edu.cn/f/0fceb9cae4c249188170/?dl=1)|[model](https://cloud.tsinghua.edu.cn/f/91cecd9ef97843339c79/?dl=1)|
| detpro + cascade | 20e | 21.6 29.8 35.1 30.5 &nbsp;&nbsp;&nbsp;| 19.8 26.8 30.3 26.9 &nbsp;&nbsp;&nbsp;|[config](https://github.com/dyabel/detpro/blob/main/configs/lvis/cascade_mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain_ens.py) | [prompt](https://cloud.tsinghua.edu.cn/f/0fceb9cae4c249188170/?dl=1)|[model](https://cloud.tsinghua.edu.cn/f/f75712011cd342bdb49e/?dl=1)  |
# Installation
This repo is built on mmdetection [mmdetection](https://github.com/open-mmlab/mmdetection) and [CoOP](https://github.com/kaiyangzhou/coop)

install mmdet2.7
```shell
pip install -r requirements/build.txt
pip install -e .
pip install git+https://github.com/openai/CLIP.git
pip uninstall pycocotools -y
pip uninstall mmpycocotools -y
pip install mmpycocotools
pip install git+https://github.com/lvis-dataset/lvis-api.git
pip install mmcv-full==1.2.5 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html


```
# Get Started
## Prepare proposal embedding, label, iou for prompt trainig
```shell
./tools/dist_train.sh  configs/lvis/prompt_save_train.py 8 --work-dir workdirs/prompt_save_train
./tools/dist_train.sh  configs/lvis/prompt_save_train.py 8 --work-dir workdirs/prompt_save_train
```
## Train prompt
```shell
cd prompt
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_5_6_end soft 0.5 0.5 0.6 8 end
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_6_7_end soft 0.5 0.6 0.7 8 end
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_7_8_end soft 0.5 0.7 0.8 8 end
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_8_9_end soft 0.5 0.8 0.9 8 end
python run.py train data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_9_10_end soft 0.5 0.9 1.1 8 end
python run.py test data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/exp fg_bg_5_10_end soft 0.0 0.5 0.5 checkpoints/exp/fg_bg_5_5_6_endepoch6.pth checkpoints/exp/fg_bg_5_6_7_endepoch6.pth checkpoints/exp/fg_bg_5_7_8_endepoch6.pth checkpoints/exp/fg_bg_5_8_9_endepoch6.pth checkpoints/exp/fg_bg_5_9_10_endepoch6.pth
./tools/dist_train.sh  configs/lvis/detpro_ens.py 8 --work-dir workdirs/vild_ens_1x_fg_bg_5_10_end --cfg-options model.roi_head.prompt_path=checkpoints/exp/fg_bg_5_10_end_ens.pth

```
## Train OVOD detector
```
#save clip image embedding
./tools/dist_train.sh  configs/lvis/detpro_ens_2x.py 8 --work-dir workdirs/vild_ens_2x_fg_bg_5_10_end --cfg-options model.roi_head.prompt_path=checkpoints/exp/fg_bg_5_10_end_ens.pth model.roi_head.load_feature=False totol_epochs=1
#zip clip image embedding
zip -r lvis_clip_image_embedding.zip data/lvis_clip_image_embedding/*
./tools/dist_train.sh  configs/lvis/detpro_ens_2x.py 8 --work-dir workdirs/vild_ens_2x_fg_bg_5_10_end --cfg-options model.roi_head.prompt_path=checkpoints/exp/fg_bg_5_10_end_ens.pth model.roi_head.load_feature=True
```
## Generate class embedding for tranfer Datasets(take objects365 as example)
```
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_5_6_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_6_7_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_7_8_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_8_9_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python gen_cls_embedding.py checkpoints/exp/fg_bg_5_9_10_endepoch6_prompt.pth checkpoints/obj365/fg_bg_5_5_6_obj365 obj365
python run.py test data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/obj365 fg_bg_5_10_obj365 soft 0.0 0.5 0.5 checkpoints/obj365/fg_bg_5_5_6_obj365.pth checkpoints/obj365/fg_bg_5_6_7_obj365.pth checkpoints/obj365/fg_bg_5_7_8_obj365.pth checkpoints/obj365/fg_bg_5_8_9_obj365.pth checkpoints/obj365/fg_bg_5_9_10_obj365.pth
```
## Transfer to other datasets
```
 ./tools/dist_test.sh  configs/transfer/transfer_voc.py workdirs/vild_ens_2x_neg5_ens/epoch_24.pth 8 --eval mAP --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/voc/fg_bg_6_10_voc_neg30_ens.pth model.roi_head.fixed_lambda=0.6
./tools/dist_test.sh  configs/transfer/transfer_coco.py workdirs/vild_ens_2x_neg5_ens/epoch_24.pth 8 --eval bbox --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/coco/fg_bg_6_10_voc_neg30_ens.pth model.roi_head.fixed_lambda=0.6
./tools/dist_test.sh  configs/transfer/transfer_objects365.py workdirs/vild_ens_2x_neg5_ens/epoch_20.pth 8 --eval bbox --cfg-options model.roi_head.load_feature=False model.roi_head.prompt_path=checkpoints/obj365/fg_bg_6_10_obj365_neg30_ens.pth model.roi_head.fixed_lambda=0.6
```
# Citation
